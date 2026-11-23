#!/usr/bin/env python3
"""GPU Pulse multi-GPU supervisor. Passive per-GPU CUPTI workers + local read-only API.

Usage: python node_agent.py --port 8766
Requires: NVIDIA CUDA driver and CUPTI for counter collection.
The API remains available if individual collectors fail.
"""
from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlsplit

if __package__:
    from .gpu_discovery import discover_gpus
    from .node_aggregator import node_summary, compute_summary
    from .peak_registry import match_peak, load_registry
    from .webui import PAGE, detail_payload, require_loopback
    from .sample_history import HISTORY_CAP, HISTORY_SECONDS, SampleHistory, average_buckets, chart_samples
else:
    from gpu_discovery import discover_gpus
    from node_aggregator import node_summary, compute_summary
    from peak_registry import match_peak, load_registry
    from webui import PAGE, detail_payload, require_loopback
    from sample_history import HISTORY_CAP, HISTORY_SECONDS, SampleHistory, average_buckets, chart_samples

LOG = logging.getLogger("gpu-pulse-node")
ROOT = Path(__file__).resolve().parent


class NodeSupervisor:
    def __init__(self, inventory, *, output_dir=None, worker_options=None, max_history=HISTORY_CAP, history_seconds=HISTORY_SECONDS):
        self.inventory = list(inventory)
        self.by_uuid = {g["uuid"]: g for g in self.inventory}
        if len(self.by_uuid) != len(self.inventory):
            raise ValueError("Duplicate GPU UUID")
        self.peak_matches = {g["uuid"]: match_peak(g) for g in self.inventory}
        self.output_dir = Path(output_dir or tempfile.mkdtemp(prefix="gpu-pulse-node-")).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.worker_options = list(worker_options or [])
        self.lock = threading.RLock()
        self.latest = {}
        self.history = {g["uuid"]: SampleHistory(history_seconds, max_history) for g in self.inventory}
        self.workers = {}
        self.threads = []

    def _reader(self, uuid, pipe):
        with pipe:
            for line in pipe:
                try:
                    data = json.loads(line)
                    if not isinstance(data, dict):
                        raise ValueError("Expected a sample object")
                    if data.get("device") != self.by_uuid[uuid]["cuda_index"]:
                        continue
                    data.setdefault("host_received_ns", time.time_ns())
                    if type(data["host_received_ns"]) is not int:
                        raise ValueError("Expected integer host timestamp")
                    with self.lock:
                        self.latest[uuid] = data
                        self.history[uuid].samples.append(data)
                        self.history[uuid].prune(time.time_ns())
                except ValueError:
                    LOG.warning("Malformed worker sample for %s", uuid)

    def start(self):
        for gpu in self.inventory:
            if gpu.get("profiling_available") is False:
                self.workers[gpu["uuid"]] = {"status": "MIG_UNSUPPORTED"}
            else:
                self._launch(gpu, delay=2)

    def _launch(self, gpu, delay):
        uuid = gpu["uuid"]
        errfile = self.output_dir / (uuid.replace("/", "_") + ".stderr.log")
        cmd = [sys.executable, str(ROOT / "observer.py"), "--device",
               str(gpu["cuda_index"]), *self.worker_options, "--output", "-"]
        record = {"status": "STARTING", "delay": delay, "started_at": time.monotonic()}
        try:
            with errfile.open("w", encoding="utf-8") as error_output:
                proc = subprocess.Popen(cmd, cwd=str(ROOT), stderr=error_output,
                                        stdout=subprocess.PIPE, text=True, encoding="utf-8",
                                        start_new_session=True)
        except OSError as exc:
            LOG.error("GPU worker failed to start (%s): %s", uuid, exc)
            record["status"] = "FAILED"
        else:
            thread = threading.Thread(target=self._reader, args=(uuid, proc.stdout),
                                      daemon=True, name="read-" + uuid[:16])
            record.update(process=proc, pid=proc.pid, thread=thread)
            self.threads.append(thread)
            thread.start()
        self.workers[uuid] = record

    def restart_failed(self):
        """Called by the agent loop. Back off independently for each failing GPU."""
        now = time.monotonic()
        for uuid, record in self.workers.items():
            proc = record.get("process")
            if proc is None:
                if record["status"] != "FAILED":
                    continue
            elif proc.poll() in (0, None) or record["thread"].is_alive():
                continue
            if "retry_at" not in record:
                if proc is not None and now - record["started_at"] >= 60:
                    record["delay"] = 2
                record["retry_at"] = now + record["delay"]
            if now >= record["retry_at"]:
                LOG.info("Restarting collector for %s", uuid)
                self.threads[:] = [thread for thread in self.threads if thread.is_alive()]
                with self.lock:
                    self.latest.pop(uuid, None)
                self._launch(self.by_uuid[uuid], delay=min(60, record["delay"] * 2))

    def worker_status(self):
        result = {}
        for uuid, rec in self.workers.items():
            proc = rec.get("process")
            returncode = proc.poll() if proc is not None else None
            status = rec["status"]
            if proc is not None:
                if returncode is None:
                    status = "RUNNING"
                else:
                    status = "STOPPED" if returncode == 0 else "FAILED"
            result[uuid] = {"status": status, "pid": rec.get("pid"), "returncode": returncode}
            if "retry_at" in rec:
                result[uuid]["retry_in_s"] = max(0, rec["retry_at"] - time.monotonic())
        return result

    def summary(self):
        with self.lock:
            latest = dict(self.latest)
        payload = node_summary(self.inventory, latest, self.peak_matches)
        statuses = self.worker_status()
        payload["collectors"] = statuses
        payload["total_memory_mib"] = sum(g.get("memory_mib", 0) for g in self.inventory)
        payload["ready_collectors"] = sum(v["status"] == "RUNNING" for v in statuses.values())
        payload["precision"] = {
            key: compute_summary(self.inventory, latest, self.peak_matches, key)
            for key in sorted({
                key for g in self.inventory
                for key in (self.peak_matches.get(g["uuid"], {}).get("spec") or {})
                    .get("compute_peaks", {})
            })
        }
        return payload

    def detail(self, uuid):
        with self.lock:
            history = self.history[uuid]
            history.prune(time.time_ns())
            latest = self.latest.get(uuid, {})
            count = len(history.samples)
        return detail_payload(self.by_uuid[uuid], self.peak_matches[uuid], latest,
                              self.worker_status().get(uuid), count)

    def history_for(self, uuid, chart=False):
        with self.lock:
            history = self.history[uuid]
            history.prune(time.time_ns())
            samples = list(history.samples)
        return {"uuid": uuid, "gpu": self.by_uuid[uuid], "alignment": "host_decode_receipt_proxy",
                "samples": chart_samples(samples) if chart else samples}

    def node_history(self):
        keys = ("dram_read_gbps", "dram_write_gbps")
        with self.lock:
            now = time.time_ns()
            for history in self.history.values():
                history.prune(now)
            histories = [list(history.samples) for history in self.history.values()]
        buckets = {}
        for samples in histories:
            for stamp, averages in average_buckets(samples, keys, require_all=True).items():
                readings = buckets.setdefault(stamp, [])
                if averages:
                    readings.append(averages)
        points = []
        for stamp, readings in sorted(buckets.items()):
            points.append({
                "timestamp_ns": stamp,
                **{key: sum(row[key] for row in readings) if readings else None for key in keys},
                "sampled_gpus": len(readings), "total_gpus": len(self.inventory),
                "status": "COMPLETE" if len(readings) == len(self.inventory) else "PARTIAL",
            })
        return {"alignment": "host_decode_receipt_proxy", "points": points}

    def shutdown(self):
        for rec in self.workers.values():
            proc = rec.get("process")
            if proc and proc.poll() is None:
                proc.terminate()
        for rec in self.workers.values():
            proc = rec.get("process")
            if proc and proc.poll() is None:
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=5)
        for thread in self.threads:
            thread.join(timeout=1)


def create_server(supervisor, host="127.0.0.1", port=8766):
    require_loopback(host)

    class Handler(BaseHTTPRequestHandler):
        def _json(self, data, status=200):
            blob = json.dumps(data, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(blob)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(blob)

        def _html(self, filename):
            raw = (ROOT / filename).read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            url = urlsplit(self.path)
            path = url.path
            if path in ("/", "/summary"):
                return self._html("node_dashboard.html")
            if path.startswith("/gpu/"):
                uuid = unquote(path[len("/gpu/"):])
                return self._html(PAGE.name) if uuid in supervisor.by_uuid else self._json({"error": "not found"}, 404)
            if path == "/api/v1/node":
                return self._json({"gpu_count": len(supervisor.inventory), "source": "CUDA Driver API"})
            if path == "/api/v1/node/summary":
                return self._json(supervisor.summary())
            if path == "/api/v1/node/history":
                return self._json(supervisor.node_history())
            if path == "/api/v1/gpus":
                return self._json({"gpus": supervisor.inventory,
                                   "collectors": supervisor.worker_status()})
            if path.startswith("/api/v1/gpus/"):
                parts = path[len("/api/v1/gpus/"):].split("/")
                uuid = unquote(parts[0])
                if uuid not in supervisor.by_uuid:
                    return self._json({"error": "GPU not found"}, 404)
                suffix = parts[1] if len(parts) == 2 else ""
                if suffix in ("", "metrics"):
                    return self._json(supervisor.detail(uuid))
                if suffix == "history":
                    return self._json(supervisor.history_for(uuid, chart=url.query == "view=chart"))
                if suffix == "capabilities":
                    match = supervisor.peak_matches[uuid]
                    return self._json({"uuid": uuid, "peak_status": match["status"],
                                       "peak_spec": match.get("spec")})
            if path.startswith("/api/v1/peak-specs/"):
                sku = unquote(path.split("/")[-1])
                for spec in load_registry():
                    if spec.get("sku") == sku:
                        return self._json(spec)
            return self._json({"error": "not found"}, 404)

        def log_message(self, fmt, *args):
            LOG.debug(fmt, *args)
    return ThreadingHTTPServer((host, port), Handler)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--history", type=int, default=HISTORY_CAP, help="Hard sample cap per GPU")
    parser.add_argument("--history-seconds", type=int, default=HISTORY_SECONDS)
    parser.add_argument("--interval-ms", type=int, default=100)
    parser.add_argument("--poll-ms", type=int, default=500)
    args = parser.parse_args(argv)
    if min(args.history, args.history_seconds) <= 0 or args.interval_ms <= 0 or args.poll_ms <= 0:
        parser.error("history and intervals must be positive")
    try:
        require_loopback(args.host)
    except ValueError as exc:
        parser.error(str(exc))
    logging.basicConfig(level=logging.INFO)
    try:
        inventory = discover_gpus()
    except (OSError, AttributeError, RuntimeError) as exc:
        parser.exit(1, f"CUDA inventory unavailable: {exc}\n")
    supervisor = NodeSupervisor(
        inventory, output_dir=args.output_dir, max_history=args.history, history_seconds=args.history_seconds,
        worker_options=["--interval-ms", str(args.interval_ms),
                        "--poll-ms", str(args.poll_ms)])
    server = None
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    try:
        supervisor.start()
        server = create_server(supervisor, args.host, args.port)
        LOG.info("GPU Pulse node: http://%s:%s/ | discovered %d GPUs", args.host, args.port, len(inventory))
        server.timeout = 0.3
        while not stop.is_set():
            server.handle_request()
            if not stop.is_set():
                supervisor.restart_failed()
    except KeyboardInterrupt:
        pass
    finally:
        if server:
            server.server_close()
        supervisor.shutdown()


if __name__ == "__main__":
    main()
