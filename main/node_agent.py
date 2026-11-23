#!/usr/bin/env python3
"""GPU Pulse multi-GPU supervisor. Passive per-GPU CUPTI workers + local read-only API.

Usage: python node_agent.py --port 8766
Requires: nvidia-smi, NVIDIA driver and CUPTI for counter collection.
The API remains available if individual collectors fail.
"""
from __future__ import annotations
import argparse
import collections
import json
import logging
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlsplit

from gpu_discovery import discover_gpus, discover_mig_mode, read_gpu_health
from node_aggregator import node_summary, compute_summary
from peak_registry import match_peak, load_registry
from webui import PAGE, detail_payload

LOG = logging.getLogger("gpu-pulse-node")
ROOT = Path(__file__).resolve().parent


class NodeSupervisor:
    def __init__(self, inventory, *, output_dir=None, worker_options=None, max_history=256):
        self.inventory = list(inventory)
        self.by_uuid = {g["uuid"]: g for g in inventory}
        if len(self.by_uuid) != len(self.inventory):
            raise ValueError("Duplicate GPU UUID")
        self.peak_matches = {g["uuid"]: match_peak(g) for g in inventory}
        self.output_dir = Path(output_dir or tempfile.mkdtemp(prefix="gpu-pulse-node-")).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.worker_options = list(worker_options or [])
        self.max_history = max_history
        self.lock = threading.RLock()
        self.latest = {}
        self.history = {g["uuid"]: collections.deque(maxlen=max_history) for g in inventory}
        self.workers = {}
        self._health_at = 0.0
        self._health_data = {}
        self.stop_event = threading.Event()
        self.threads = []

    def _reader(self, uuid, path, proc):
        while not self.stop_event.is_set() and not path.exists() and proc.poll() is None:
            self.stop_event.wait(0.1)
        if not path.exists():
            return
        with path.open(encoding="utf-8") as fp:
            while not self.stop_event.is_set():
                line = fp.readline()
                if not line:
                    if proc.poll() is not None:
                        break
                    self.stop_event.wait(0.15)
                    continue
                try:
                    data = json.loads(line)
                    if data.get("device") != self.by_uuid[uuid]["cuda_index"]:
                        continue
                    # Host decode time is a coarse cross-GPU alignment proxy,
                    # NOT a synchronized GPU hardware timestamp.
                    data.setdefault("host_received_ns", time.time_ns())
                    with self.lock:
                        self.latest[uuid] = data
                        self.history[uuid].append(data)
                except (ValueError, TypeError, KeyError):
                    LOG.warning("Malformed worker sample for %s", uuid)

    def start(self):
        for gpu in self.inventory:
            uuid = gpu["uuid"]
            if gpu.get("cuda_index") is None:
                self.workers[uuid] = {"status": "UNMAPPED", "pid": None}
                continue
            if gpu.get("mig_mode") == "Enabled":
                self.workers[uuid] = {"status": "MIG_UNSUPPORTED", "pid": None}
                continue
            path = self.output_dir / (uuid.replace("/", "_") + ".jsonl")
            # New file each run to avoid stale output at restart.
            path.write_text("", encoding="utf-8")
            errfile = self.output_dir / (uuid.replace("/", "_") + ".stderr.log")
            cmd = [sys.executable, str(ROOT / "observer.py"), "--device",
                   str(gpu["cuda_index"]), "--output", str(path), *self.worker_options]
            try:
                with errfile.open("w", encoding="utf-8") as error_output:
                    proc = subprocess.Popen(cmd, cwd=str(ROOT), stderr=error_output,
                                            stdout=subprocess.DEVNULL, start_new_session=True)
            except OSError as exc:
                LOG.error("GPU worker failed to start (%s): %s", uuid, exc)
                self.workers[uuid] = {"status": "FAILED", "pid": None,
                                      "error_log": str(errfile), "reason": str(exc)}
                continue
            self.workers[uuid] = {"status": "STARTING", "pid": proc.pid, "process": proc,
                                  "error_log": str(errfile)}
            t = threading.Thread(target=self._reader, args=(uuid, path, proc),
                                 daemon=True, name="read-" + uuid[:16])
            t.start()
            self.threads.append(t)

    def worker_status(self):
        result = {}
        for uuid, rec in self.workers.items():
            proc = rec.get("process")
            status = ("RUNNING" if proc and proc.poll() is None else
                      "FAILED" if proc and proc.poll() is not None else rec["status"])
            result[uuid] = {"status": status, "pid": rec.get("pid"),
                            "returncode": proc.returncode if proc else None,
                            "error_log": rec.get("error_log")}
        return result

    def snapshot(self):
        with self.lock:
            latest = dict(self.latest)
            history = {k: list(v) for k, v in self.history.items()}
        return latest, history

    def health_snapshot(self):
        with self.lock:
            if time.monotonic() - self._health_at < 2:
                return dict(self._health_data)
        # Do not hold the sample ring-buffer lock during a subprocess call.
        health = read_gpu_health()
        with self.lock:
            self._health_data = health
            self._health_at = time.monotonic()
            return dict(self._health_data)

    def summary(self):
        latest, _ = self.snapshot()
        payload = node_summary(self.inventory, latest, self.peak_matches)
        statuses = self.worker_status()
        payload["collectors"] = statuses
        payload["total_memory_mib"] = sum(g.get("memory_mib", 0) for g in self.inventory)
        health = self.health_snapshot()
        for gpu in payload["gpus"]:
            gpu["health"] = health.get(gpu["uuid"], {})
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
        gpu = self.by_uuid[uuid]
        latest, history = self.snapshot()
        return detail_payload(
            gpu, self.peak_matches[uuid], latest.get(uuid, {}),
            self.worker_status().get(uuid), self.health_snapshot().get(uuid, {}),
            len(history[uuid]),
        )

    def history_for(self, uuid):
        _, history = self.snapshot()
        return {"uuid": uuid, "alignment": "host_decode_receipt_proxy",
                "samples": history[uuid]}

    def node_history(self):
        """Arrival-time bins; label PARTIAL when any GPU is missing from a bin."""
        _, histories = self.snapshot()
        buckets = {}
        for uid, records in histories.items():
            for record in records:
                stamp = record.get("host_received_ns")
                if not isinstance(stamp, int):
                    continue
                second = stamp // 1_000_000_000
                b = buckets.setdefault(second, {})
                b[uid] = record
        points = []
        for second, b in sorted(buckets.items())[-self.max_history:]:
            measured = []
            for rec in b.values():
                m = rec.get("measurements", {})
                r, w = m.get("dram_read_gbps"), m.get("dram_write_gbps")
                if isinstance(r, (float, int)) and isinstance(w, (float, int)):
                    measured.append((r, w))
            points.append({
                "timestamp_ns": second * 1_000_000_000,
                "dram_read_gbps": sum(x[0] for x in measured) if measured else None,
                "dram_write_gbps": sum(x[1] for x in measured) if measured else None,
                "sampled_gpus": len(measured), "total_gpus": len(self.inventory),
                "status": "COMPLETE" if len(measured) == len(self.inventory) else "PARTIAL",
            })
        return {"alignment": "host_decode_receipt_proxy", "points": points}

    def shutdown(self):
        self.stop_event.set()
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
            path = urlsplit(self.path).path
            if path in ("/", "/summary"):
                return self._html("node_dashboard.html")
            if path.startswith("/gpu/"):
                uuid = unquote(path[len("/gpu/"):])
                return self._html(PAGE.name) if uuid in supervisor.by_uuid else self._json({"error": "not found"}, 404)
            if path == "/api/v1/node":
                return self._json({"gpu_count": len(supervisor.inventory), "source": "nvidia-smi/CUDA"})
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
                    return self._json(supervisor.history_for(uuid))
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
    parser.add_argument("--history", type=int, default=256)
    parser.add_argument("--interval-ms", type=int, default=100)
    parser.add_argument("--poll-ms", type=int, default=500)
    args = parser.parse_args(argv)
    if args.history <= 0 or args.interval_ms <= 0 or args.poll_ms <= 0:
        parser.error("history and intervals must be positive")
    logging.basicConfig(level=logging.INFO)
    inventory = discover_gpus()
    mig = discover_mig_mode()
    for gpu in inventory:
        gpu["mig_mode"] = mig.get(gpu["nvml_index"], "Unknown")
    supervisor = NodeSupervisor(
        inventory, output_dir=args.output_dir, max_history=args.history,
        worker_options=["--interval-ms", str(args.interval_ms),
                        "--poll-ms", str(args.poll_ms)])
    server = None
    try:
        supervisor.start()
        server = create_server(supervisor, args.host, args.port)
        LOG.info("GPU Pulse node: http://%s:%s/ | discovered %d GPUs", args.host, args.port, len(inventory))
        server.serve_forever(poll_interval=0.3)
    except KeyboardInterrupt:
        pass
    finally:
        if server:
            server.server_close()
        supervisor.shutdown()


if __name__ == "__main__":
    main()
