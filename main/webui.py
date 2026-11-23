"""Shared GPU Detail payload and standalone read-only HTTP server."""
from __future__ import annotations
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit
import json
import threading

if __package__:
    from .sample_history import chart_samples
    from .gpu_discovery import discover_gpus
    from .peak_registry import match_peak, measured_compute
else:
    from sample_history import chart_samples
    from gpu_discovery import discover_gpus
    from peak_registry import match_peak, measured_compute

PAGE = Path(__file__).with_name("gpu_detail.html")


def detail_payload(gpu, match, sample, collector, history_count):
    spec = match.get("spec") or {}
    compute = measured_compute(sample.get("precision_metrics", {}), spec)
    for key, reference in spec.get("compute_peaks", {}).items():
        compute.setdefault(key, {"measured": None, "peak": reference["value"],
                                 "unit": reference["unit"], "attainment_pct": None})
    for key, record in compute.items():
        reference = spec.get("compute_peaks", {}).get(key, {})
        record["peak_origin"] = reference.get("value_origin")
        if reference.get("value_origin") == "AMBIGUOUS_SOURCE":
            record["peak"] = None
            record["attainment_pct"] = None
        if record["attainment_pct"] is not None:
            record["comparison_note"] = "Dense theoretical reference; runtime sparsity not independently verified"
    return {
        "gpu": gpu, "collector": collector,
        "peak_status": match["status"], "sku": match.get("sku"),
        "peak_source": spec.get("source"),
        "memory_bandwidth_peak_gbps": (None if spec.get("memory_peak_quality") == "LOWER_BOUND"
                                       else spec.get("memory_bandwidth_gbps")),
        "compute": compute, "measurements": sample.get("measurements", {}),
        "raw_metrics": sample.get("raw_metrics", {}),
        "last_host_received_ns": sample.get("host_received_ns"),
        "history_count": history_count,
    }


def require_loopback(host):
    if host != "127.0.0.1":
        raise ValueError("Dashboard binds to 127.0.0.1 only. Use SSH forwarding for remote access.")


def start_server(state, host: str = "127.0.0.1", port: int = 8765):
    require_loopback(host)
    device = state.snapshot()["status"].get("device")
    try:
        gpu = next((g for g in discover_gpus() if device is not None and g["cuda_index"] == device), None)
    except (OSError, AttributeError, RuntimeError):
        gpu = None
    gpu = gpu or {"cuda_index": device}
    match = match_peak(gpu)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            url = urlsplit(self.path)
            path = url.path
            if path in ("/api/state", "/api/detail"):
                state_snapshot = state.snapshot()
                if path == "/api/detail":
                    samples = state_snapshot["samples"]
                    collector = dict(state_snapshot["status"])
                    collector["status"] = collector.pop("state", "unknown").upper()
                    state_snapshot = detail_payload(
                        gpu, match, samples[-1] if samples else {}, collector, len(samples)
                    )
                    state_snapshot["samples"] = chart_samples(samples) if url.query == "view=chart" else samples
                payload = json.dumps(state_snapshot, allow_nan=False).encode()
                content_type = "application/json; charset=utf-8"
            elif path in ("/", "/dashboard.html"):
                payload = PAGE.read_bytes()
                content_type = "text/html; charset=utf-8"
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer((host, port), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True, name="observer-web")
    thread.start()
    return server
