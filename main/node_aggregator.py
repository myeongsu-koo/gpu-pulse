"""Conservative multi-GPU snapshot aggregation. Missing GPUs are never zero-filled."""
from __future__ import annotations
import math
import time
from peak_registry import attain


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


def node_summary(inventory, latest, peak_matches=None, now_ns=None, max_age_s=5.0, max_skew_s=2.0):
    """Aggregate fresh host-arrival snapshots, not unrelated GPU clock domains.

    The observer's host_received_ns is a decode-time proxy, not exact sample time.
    A node-wide point is COMPLETE only if each CUDA-visible device has fresh values
    and the receipt-time spread is within max_skew_s.
    """
    peak_matches = peak_matches or {}
    now_ns = time.time_ns() if now_ns is None else now_ns
    rows = []
    total_read = total_write = 0.0
    complete_mem = []
    timestamps = []
    for gpu in inventory:
        uid = gpu["uuid"]
        sample = latest.get(uid)
        stamp = sample.get("host_received_ns") if sample else None
        fresh = isinstance(stamp, int) and 0 <= now_ns - stamp <= max_age_s * 1e9
        if fresh:
            timestamps.append(stamp)
        measurements = sample.get("measurements", {}) if fresh else {}
        read, write = measurements.get("dram_read_gbps"), measurements.get("dram_write_gbps")
        valid_mem = _number(read) and _number(write)
        if valid_mem:
            total_read += read
            total_write += write
            complete_mem.append(uid)
        peak_match = peak_matches.get(uid, {})
        spec = peak_match.get("spec")
        peak_bw = (spec or {}).get("memory_bandwidth_gbps")
        if (spec or {}).get("memory_peak_quality") == "LOWER_BOUND":
            peak_bw = None
        mem_pct = attain(read + write, peak_bw) if valid_mem else None
        row = {"uuid": uid, "cuda_index": gpu.get("cuda_index"), "model": gpu["model"],
               "pci_bus_id": gpu.get("pci_bus_id"), "status": "FRESH" if fresh else "STALE",
               "sample_age_s": (now_ns - stamp) / 1e9 if fresh else None,
               "dram_read_gbps": read if valid_mem else None,
               "dram_write_gbps": write if valid_mem else None,
               "dram_attainment_pct": mem_pct,
               "peak_status": peak_match.get("status", "SPEC_UNAVAILABLE"),
               "sku": peak_match.get("sku"),
               "measurements": measurements}
        rows.append(row)
    skew = (max(timestamps) - min(timestamps)) / 1e9 if timestamps else None
    counted = len(complete_mem)
    all_counted = counted == len(inventory) and len(inventory) > 0
    aligned = skew is not None and skew <= max_skew_s
    # No aggregate of percentages; all GPU bandwidths refer to same unit (decimal GB/s).
    peak_total = 0.0
    peak_known = True
    for uid in complete_mem:
        spec = peak_matches.get(uid, {}).get("spec") or {}
        peak = spec.get("memory_bandwidth_gbps")
        if spec.get("memory_peak_quality") == "LOWER_BOUND":
            peak = None
        if not _number(peak) or peak == 0:
            peak_known = False
            break
        peak_total += peak
    partial_pct = attain(total_read + total_write, peak_total) if peak_known and peak_total else None
    return {
        "gpu_count": len(inventory), "sampled_memory_gpu_count": counted,
        "coverage_pct": 100.0 * counted / len(inventory) if inventory else 0.0,
        "aggregation_status": "COMPLETE" if all_counted and aligned else "PARTIAL",
        "alignment": "host_decode_receipt_proxy", "max_sample_skew_s": skew,
        "dram_read_gbps": total_read if counted and aligned else None,
        "dram_write_gbps": total_write if counted and aligned else None,
        "dram_attainment_pct": partial_pct if aligned else None,
        "gpus": rows,
    }


def compute_summary(inventory, latest, peak_matches, metric_key, now_ns=None, max_age_s=5.0):
    """Sum only equivalent precision/path metrics with matching per-SKU peaks."""
    now_ns = time.time_ns() if now_ns is None else now_ns
    total, peak, counted = 0.0, 0.0, 0
    timestamps = []
    for gpu in inventory:
        uid = gpu["uuid"]
        sample = latest.get(uid, {})
        stamp = sample.get("host_received_ns")
        if not isinstance(stamp, int) or not 0 <= now_ns-stamp <= max_age_s*1e9:
            continue
        metrics = sample.get("precision_metrics", {})
        value = (metrics.get(metric_key) or {}).get("value")
        spec = (peak_matches.get(uid, {}) or {}).get("spec") or {}
        peak_rec = spec.get("compute_peaks", {}).get(metric_key)
        if not _number(value) or peak_rec is None or not _number(peak_rec.get("value")):
            continue
        total += value
        peak += peak_rec["value"]
        counted += 1
        timestamps.append(stamp)
    aligned = bool(timestamps) and (max(timestamps)-min(timestamps))/1e9 <= 2
    return {
        "metric": metric_key, "measured": total if counted and aligned else None,
        "peak": peak if counted and aligned else None,
        "attainment_pct": attain(total, peak) if counted and aligned else None,
        "status": "COMPLETE" if counted == len(inventory) and aligned and counted else "PARTIAL",
        "counted_gpus": counted, "total_gpus": len(inventory),
    }
