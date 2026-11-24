#!/usr/bin/env python3
"""CUPTI PM Sampling GPU observer: hardware-counter time series, never inferred FLOPs.

Requires Linux NVIDIA driver, CUPTI Python >=13.2 and profiling privileges.
Outputs NDJSON; optional read-only local HTTP dashboard.
"""
from __future__ import annotations

import argparse
import contextlib
import ctypes
import json
import logging
import os
import signal
import sys
import threading
import time
from pathlib import Path

if __package__:
    from .sample_schema import nonnegative_number
    from .collector_config import DEFAULT_INTERVAL_MS, PROFILER_BUSY_EXIT, profiler_busy
    from .gpu_discovery import discover_gpus
    from .sample_history import HISTORY_CAP, HISTORY_SECONDS, SampleHistory
    from .metrics import SPECS, KEY_TO_SPEC, choose_set, coverage, select_for_set, to_dashboard_values, precision_dashboard_values
else:
    from sample_schema import nonnegative_number
    from collector_config import DEFAULT_INTERVAL_MS, PROFILER_BUSY_EXIT, profiler_busy
    from gpu_discovery import discover_gpus
    from sample_history import HISTORY_CAP, HISTORY_SECONDS, SampleHistory
    from metrics import SPECS, KEY_TO_SPEC, choose_set, coverage, select_for_set, to_dashboard_values, precision_dashboard_values

LOG = logging.getLogger("cupti-observer")


class CudaPrimaryContext:
    """Initialize just enough CUDA to satisfy Collector.enable().

    Uses libcuda directly so no PyTorch, Numba, synthetic kernel, or runtime
    process instrumentation is required. Does not synchronize other processes.
    """

    def __init__(self, index: int):
        self.index = index
        self.lib = None
        self.dev = ctypes.c_int()
        self.ctx = ctypes.c_void_p()
        self.retained = False

    @staticmethod
    def check(code: int, call: str) -> None:
        if code != 0:
            raise RuntimeError(f"CUDA Driver {call} returned {code}. Check driver/device visibility.")

    def __enter__(self):
        self.lib = ctypes.CDLL("libcuda.so.1")
        self.lib.cuInit.argtypes = [ctypes.c_uint]
        self.lib.cuInit.restype = ctypes.c_int
        self.lib.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        self.lib.cuDeviceGet.restype = ctypes.c_int
        self.lib.cuDevicePrimaryCtxRetain.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int]
        self.lib.cuDevicePrimaryCtxRetain.restype = ctypes.c_int
        self.lib.cuDevicePrimaryCtxRelease.argtypes = [ctypes.c_int]
        self.lib.cuDevicePrimaryCtxRelease.restype = ctypes.c_int
        self.lib.cuCtxSetCurrent.argtypes = [ctypes.c_void_p]
        self.lib.cuCtxSetCurrent.restype = ctypes.c_int
        self.check(self.lib.cuInit(0), "cuInit")
        self.check(self.lib.cuDeviceGet(ctypes.byref(self.dev), self.index), "cuDeviceGet")
        self.check(self.lib.cuDevicePrimaryCtxRetain(ctypes.byref(self.ctx), self.dev.value), "cuDevicePrimaryCtxRetain")
        self.retained = True
        self.check(self.lib.cuCtxSetCurrent(self.ctx), "cuCtxSetCurrent")
        return self

    def __exit__(self, *_):
        if self.lib and self.retained:
            self.lib.cuCtxSetCurrent(None)
            self.lib.cuDevicePrimaryCtxRelease(self.dev.value)
            self.retained = False


def get_chip_name(device_index: int) -> str:
    """Documented CUPTI DeviceGetChipName via low-level cupti-python binding."""
    from cupti import cupti as low
    p = low.Device_GetChipName_Params()
    p.struct_size = low.DEVICE_GET_CHIP_NAME_PARAMS_STRUCT_SIZE
    p.device_index = device_index
    low.device_get_chip_name(p.ptr)
    name = p.p_chip_name
    return name.decode() if isinstance(name, bytes) else str(name)


def discover_sets(chip: str) -> dict[str, list[str]]:
    from cupti import profiler_host
    return {name: profiler_host.get_metrics_in_single_pass_set(chip, name)
            for name in profiler_host.get_single_pass_sets(chip)}


def select_plan(chip: str, forced_set: str | None, required: list[str]) -> tuple[str, list]:
    sets = discover_sets(chip)
    if not sets:
        raise RuntimeError("CUPTI did not expose single-pass metric sets; requires supported CUPTI 13.2+.")
    if forced_set is not None:
        if forced_set not in sets:
            raise ValueError(f"Unknown single-pass set {forced_set!r}; available: {list(sets)}")
        specs = select_for_set(sets[forced_set])
        missing = set(required) - {s.key for s in specs}
        if missing:
            raise ValueError(f"Requested metric(s) absent from {forced_set}: {sorted(missing)}")
        if not specs:
            raise ValueError(f"{forced_set}: none of the supported target metrics are present")
        return forced_set, specs
    return choose_set(sets, required)


def capability_report(chip: str) -> dict:
    sets = discover_sets(chip)
    keys_by_set = {name: {s.key for s in select_for_set(metrics)}
                   for name, metrics in sets.items()}
    preferred = {}
    for s in SPECS:
        preferred[s.key] = {"requested_metric": s.name,
                            "sets_with_base": [name for name, keys in keys_by_set.items()
                                               if s.key in keys]}
    try:
        best_set, best_specs = choose_set(sets)
        best = {"set": best_set, "metric_names": [s.name for s in best_specs], "coverage": coverage(best_specs)}
    except ValueError:
        best = None
    return {"chip": chip, "single_pass_set_count": len(sets), "candidate_availability": preferred,
            "preferred_plan": best,
            "note": "base presence is a preliminary check; Collector.configure() decides actual support"}


class State:
    def __init__(self, capacity=HISTORY_CAP, seconds=HISTORY_SECONDS):
        self.lock = threading.Lock()
        self.history = SampleHistory(seconds, capacity)
        self.status = {"state": "initializing", "metrics": [], "coverage": {}, "chip": None}

    def set_status(self, **kwargs):
        with self.lock:
            self.status.update(kwargs)

    def push(self, record):
        with self.lock:
            self.history.samples.append(record)
            self.history.prune(record["host_received_ns"])

    def snapshot(self):
        with self.lock:
            self.history.prune(time.time_ns())
            return {"status": dict(self.status), "samples": list(self.history.samples)}


def emit_sample(sample, names: list[str], selected, chip: str, device: int) -> dict:
    raw = {name: nonnegative_number(value) for name, value in zip(names, sample.metric_values, strict=True)}
    ts_start, ts_end = int(sample.start_timestamp), int(sample.end_timestamp)
    return {
        "chip": chip,
        "device": device,
        "gpu_start_ns": ts_start,
        "gpu_end_ns": ts_end,
        "host_received_ns": time.time_ns(),
        "sample_duration_ms": max(0, ts_end - ts_start) / 1e6,
        "measurements": to_dashboard_values(
            raw, selected, (ts_end - ts_start) / 1e9 if ts_end > ts_start else None
        ),
        "raw_metrics": raw,
        "precision_metrics": precision_dashboard_values(raw, selected),
        "metric_coverage": coverage(selected)["tensor_precisions"],
    }


def run(args) -> int:
    try:
        from cupti import pm_sampling
    except ImportError as ex:
        raise RuntimeError("CUPTI Python missing. Install with `pip install 'cupti-python>=13.2,<14'` on GPU Linux host.") from ex

    stop = threading.Event()
    def handle_stop(_signum, _frame):
        stop.set()
    signal.signal(signal.SIGINT, handle_stop)
    signal.signal(signal.SIGTERM, handle_stop)

    state = State(seconds=args.history_seconds)
    with CudaPrimaryContext(args.device):
        chip = get_chip_name(args.device)
        if args.probe:
            print(json.dumps(capability_report(chip), indent=2, ensure_ascii=False))
            return 0
        inventory = {}
        if args.web or (args.output and not getattr(args, "worker_stream", False)):
            try:
                inventory = next((gpu for gpu in discover_gpus() if gpu["cuda_index"] == args.device), {})
            except (OSError, AttributeError, RuntimeError) as exc:
                LOG.warning("CUDA inventory unavailable: %s", exc)
        requested = [r.strip() for r in args.require.split(",") if r.strip()]
        unknown = set(requested) - set(KEY_TO_SPEC)
        if unknown:
            raise ValueError(f"Unknown --require keys: {sorted(unknown)}. Valid: {list(KEY_TO_SPEC)}")
        set_name, specs = select_plan(chip, args.metric_set, requested)
        names = [s.name for s in specs]
        is_ga100 = chip.lower().startswith("ga100")
        interval = args.sysclk_cycles if is_ga100 else args.interval_ms * 1_000_000
        trigger = (pm_sampling.TriggerMode.GPU_SYSCLK_INTERVAL if is_ga100
                   else pm_sampling.TriggerMode.GPU_TIME_INTERVAL)
        state.set_status(state="configured", device=args.device, chip=chip,
                         single_pass_set=set_name, metrics=names, coverage=coverage(specs),
                         trigger=trigger.name, sampling_interval=int(interval))
        LOG.info("GPU %d (%s), set=%s, metrics=%s", args.device, chip, set_name, ", ".join(names))

        collector = pm_sampling.Collector(device_index=args.device)
        collector.enable()
        started = False
        file = None
        server = None
        try:
            collector.configure(
                metrics=names,
                hardware_buffer_size=args.hardware_buffer_mb * 1024 * 1024,
                sampling_interval=int(interval),
                trigger_mode=trigger,
                single_pass_metric_set_name=set_name,
                hw_buffer_append_mode=pm_sampling.HardwareBufferAppendMode.KEEP_LATEST,
            )
            if args.web:
                if __package__:
                    from .webui import start_server
                else:
                    from webui import start_server
                server = start_server(state, host=args.bind, port=args.port, gpu=inventory,
                                      benchmark_results=args.benchmark_results)
                LOG.info("Dashboard: http://%s:%d/", args.bind, args.port)
            if args.output == "-":
                file = sys.stdout
            elif args.output:
                file = open(args.output, "a", encoding="utf-8", buffering=1)
            LOG.info("CUPTI PM Sampling started; no workload is launched by this observer")
            session_start_ns = time.monotonic_ns()
            collector.start()
            started = True
            state.set_status(state="running")
            time_start = time.monotonic()
            prev_end = -1

            def decode_batch():
                nonlocal prev_end
                batch = collector.decode(max_samples=args.max_samples)
                session_elapsed_ns = time.monotonic_ns() - session_start_ns
                decoded = 0
                for sample in batch:
                    decoded += 1
                    duration_ns = sample.end_timestamp - sample.start_timestamp
                    # A GPU sample window cannot span more than the host sampling session.
                    if sample.end_timestamp <= prev_end or not 0 < duration_ns <= session_elapsed_ns:
                        continue
                    prev_end = sample.end_timestamp
                    record = emit_sample(sample, batch.metrics, specs, chip, args.device)
                    if inventory:
                        record["gpu"] = inventory
                    if args.web:
                        state.push(record)
                    if file:
                        file.write(json.dumps(record, allow_nan=False) + "\n")
                if file is sys.stdout:
                    file.flush()
                return decoded

            while not stop.wait(args.poll_ms / 1000):
                if args.duration > 0 and time.monotonic() - time_start >= args.duration:
                    break
                try:
                    decode_batch()
                except MemoryError:
                    LOG.warning("CUPTI HW overflow; data lost. Increase --interval-ms / --sysclk-cycles or buffer")
                    state.set_status(last_error="hardware buffer overflow; some samples may be lost")
            LOG.info("Stopping PM sampling")
            collector.stop()
            started = False
            try:
                while decode_batch():
                    pass
            except (MemoryError, RuntimeError) as ex:
                LOG.warning("Final decode incomplete: %s", ex)
            state.set_status(state="stopped")
        finally:
            if started:
                try:
                    collector.stop()
                except Exception:
                    LOG.exception("Collector stop failed")
            try:
                collector.disable()
            except Exception:
                LOG.exception("Collector disable failed")
            if file and file is not sys.stdout:
                file.close()
            if server:
                server.shutdown()
                server.server_close()
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", type=int, default=0, help="CUDA device index")
    p.add_argument("--probe", action="store_true", help="Only report metric capabilities/single-pass coverage")
    p.add_argument("--metric-set", help="Force a named CUPTI single-pass metric set")
    p.add_argument("--require", default="", help="Comma-separated internal keys required in selected single-pass set")
    p.add_argument("--interval-ms", type=int, default=DEFAULT_INTERVAL_MS, help="GPU time interval in milliseconds (except SYSCLK sampling)")
    p.add_argument("--sysclk-cycles", type=int, default=10_000_000, help="A100/GA100 interval in SYSCLK cycles")
    p.add_argument("--poll-ms", type=int, default=500, help="Host decoding/publishing interval")
    p.add_argument("--max-samples", type=int, default=4096, help="Maximum samples processed per decode() call")
    p.add_argument("--hardware-buffer-mb", type=int, default=128, help="PM hardware buffer allocation in MiB")
    p.add_argument("--duration", type=float, default=0, help="Run seconds, 0 means until Ctrl+C")
    p.add_argument("--worker-stream", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--output", default="", help="Opt-in append-only NDJSON path; - streams to stdout")
    p.add_argument("--history-seconds", type=int, default=HISTORY_SECONDS)
    p.add_argument("--web", action="store_true", help="Serve live dashboard (127.0.0.1 only by default)")
    p.add_argument("--benchmark-results", type=Path, metavar="JSON",
                   help="Show measured GEMM rates from a GPU Pulse benchmark report")
    p.add_argument("--bind", choices=["127.0.0.1"], default="127.0.0.1",
                   help="Compatibility option. Dashboard accepts IPv4 loopback only.")
    p.add_argument("--port", type=int, default=8765)
    args = p.parse_args(argv)
    if args.benchmark_results is not None and not args.web:
        p.error("--benchmark-results requires --web")
    if min(args.interval_ms, args.sysclk_cycles, args.poll_ms, args.max_samples, args.hardware_buffer_mb, args.history_seconds) <= 0:
        p.error("sampling interval, poll interval, max samples, hardware buffer must all be positive")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stderr)
    try:
        return run(args)
    except Exception as e:
        if profiler_busy(e):
            LOG.error("Another profiler is using the GPU performance counters: %s", e)
            return PROFILER_BUSY_EXIT
        LOG.error("Collector failed: %s", e)
        LOG.error("Check GPU privilege, CUPTI version, metric-set availability, and DCGM profiling conflicts")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
