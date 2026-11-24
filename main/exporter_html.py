#!/usr/bin/env python3
"""Render a standalone GPU Pulse HTML report from a local JSON or NDJSON capture."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

if __package__:
    from .sample_schema import decode_json, validate_sample
    from .sample_history import chart_samples
    from .peak_registry import match_peak
    from .webui import PAGE, detail_payload
else:
    from sample_schema import decode_json, validate_sample
    from sample_history import chart_samples
    from peak_registry import match_peak
    from webui import PAGE, detail_payload

DATA_TAG = '<script type="application/json" id="report-data">'


def read_capture(path):
    with path.open(encoding="utf-8-sig") as stream:
        try:
            capture = decode_json(stream.read())
        except json.JSONDecodeError:
            stream.seek(0)
            capture = []
            for number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    capture.append(decode_json(line))
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid NDJSON on line {number}: {error.msg}") from error

    if isinstance(capture, dict):
        if "samples" in capture:
            return capture
        return {"samples": [capture]}
    if isinstance(capture, list):
        return {"samples": capture}
    raise ValueError("Expected GPU Pulse samples or a capture with a samples array")


def render_report(capture, model=None, memory_mib=None, compact=False):
    samples = capture["samples"]
    if not isinstance(samples, list):
        raise ValueError("Expected a samples array")
    for row in samples:
        validate_sample(row)
    devices = {row["device"] for row in samples if isinstance(row.get("device"), int)}
    if len(devices) > 1:
        raise ValueError("Export samples for one CUDA device at a time")

    latest = samples[-1] if samples else {}
    inventories = [row["gpu"] for row in samples if isinstance(row.get("gpu"), dict)]
    uuids = {gpu["uuid"] for gpu in inventories if gpu.get("uuid")}
    if len(uuids) > 1:
        raise ValueError("Export samples for one GPU UUID at a time")
    gpu = dict(capture.get("gpu") or (inventories[-1] if inventories else {}))
    if model:
        gpu["model"] = model
    if memory_mib is not None:
        gpu["memory_mib"] = memory_mib
    gpu.setdefault("cuda_index", latest.get("device"))
    collector = dict(capture.get("collector") or capture.get("status") or {})
    collector["status"] = "OFFLINE REPORT"
    collector.setdefault("chip", latest.get("chip"))
    data = detail_payload(gpu, match_peak(gpu), latest, collector, len(samples))
    if compact:
        # Legacy captures may carry only device timestamps.
        points = [{**row, "host_received_ns": row.get("host_received_ns", row.get("gpu_end_ns"))}
                  for row in samples]
        data["samples"] = chart_samples(points)
    else:
        data["samples"] = samples
    # Escape HTML delimiters before inserting untrusted capture text into a script element.
    payload = json.dumps(data, allow_nan=False).replace("<", r"\u003c")
    return PAGE.read_text(encoding="utf-8").replace(DATA_TAG, DATA_TAG + payload, 1)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_path", type=Path, help="Local observer JSONL or GPU Pulse JSON capture")
    parser.add_argument("-o", "--output", type=Path, default=Path("gpu_pulse_report.html"))
    parser.add_argument("--model", help="GPU model when the capture has no inventory metadata")
    parser.add_argument("--memory-mib", type=int, help="Total GPU memory in MiB")
    parser.add_argument("--compact", action="store_true", help="Embed half-second chart averages instead of raw samples")
    args = parser.parse_args(argv)
    try:
        if args.input_path.resolve() == args.output.resolve():
            raise ValueError("Choose a different output path from the capture")
        if args.memory_mib is not None and args.memory_mib <= 0:
            raise ValueError("Total GPU memory must be positive")
        report = render_report(read_capture(args.input_path), args.model, args.memory_mib, args.compact)
        args.output.write_text(report, encoding="utf-8")
    except (OSError, ValueError, TypeError) as error:
        parser.exit(1, f"{parser.prog}: {error}\n")


if __name__ == "__main__":
    main()
