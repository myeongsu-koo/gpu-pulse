"""Conservative multi-GPU snapshot aggregation. Missing GPUs are never zero-filled."""
from __future__ import annotations
import time
if __package__:
    from .peak_registry import attain, compare_compute, memory_peak
    from .sample_schema import MAX_SAMPLE_AGE_S, nonnegative_number, sample_age_seconds
else:
    from peak_registry import attain, compare_compute, memory_peak
    from sample_schema import MAX_SAMPLE_AGE_S, nonnegative_number, sample_age_seconds


def node_summary(inventory, latest, peak_matches=None, now_ns=None, max_age_s=MAX_SAMPLE_AGE_S, max_skew_s=2.0):
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
        sample = latest.get(uid, {})
        age = sample_age_seconds(sample, now_ns)
        fresh = age is not None and age <= max_age_s
        measurements = sample.get("measurements", {}) if fresh else {}
        read, write = measurements.get("dram_read_gbps"), measurements.get("dram_write_gbps")
        valid_mem = all(nonnegative_number(value) is not None for value in (read, write))
        if valid_mem:
            total_read += read
            total_write += write
            complete_mem.append(uid)
            timestamps.append(sample["host_received_ns"])
        peak_match = peak_matches.get(uid, {})
        spec = peak_match.get("spec")
        peak_bw = memory_peak(spec)
        mem_pct = attain(read + write, peak_bw) if valid_mem else None
        row = {"uuid": uid, "cuda_index": gpu.get("cuda_index"), "model": gpu["model"],
               "pci_bus_id": gpu.get("pci_bus_id"), "status": "FRESH" if fresh else "STALE",
               "sample_age_s": age,
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
        peak = memory_peak(spec)
        if peak is None:
            peak_known = False
            break
        peak_total += peak
    partial_pct = attain(total_read + total_write, peak_total) if peak_known and peak_total else None
    return {
        "gpu_count": len(inventory), "sampled_memory_gpu_count": counted,
        "coverage_pct": 100.0 * counted / len(inventory) if inventory else 0.0,
        "aggregation_status": "COMPLETE" if all_counted and aligned else "PARTIAL",
        "alignment": "host_decode_receipt_proxy", "max_sample_skew_s": skew,
        "dram_read_gbps": nonnegative_number(total_read) if counted and aligned else None,
        "dram_write_gbps": nonnegative_number(total_write) if counted and aligned else None,
        "dram_attainment_pct": partial_pct if aligned else None,
        "gpus": rows,
    }


def compute_summary(inventory, latest, peak_matches, metric_key, now_ns=None, max_age_s=MAX_SAMPLE_AGE_S):
    """Sum only equivalent precision/path metrics with matching per-SKU peaks."""
    now_ns = time.time_ns() if now_ns is None else now_ns
    total, peak, counted = 0.0, 0.0, 0
    timestamps = []
    units = set()
    peaks_complete = True
    for gpu in inventory:
        uid = gpu["uuid"]
        sample = latest.get(uid, {})
        age = sample_age_seconds(sample, now_ns)
        if age is None or age > max_age_s:
            continue
        reading = sample.get("precision_metrics", {}).get(metric_key)
        spec = peak_matches.get(uid, {}).get("spec") or {}
        comparison = compare_compute(metric_key, reading, spec.get("compute_peaks", {}).get(metric_key))
        unit = comparison["measured_unit"]
        if (comparison["measured"] is None or not comparison["counter_unit_verified"]
                or unit not in ("TFLOP/s", "TOP/s")
                or (comparison["peak"] is not None and unit != comparison["unit"])):
            continue
        peaks_complete = peaks_complete and comparison["attainment_pct"] is not None
        total += comparison["measured"]
        peak += comparison["peak"] or 0
        counted += 1
        units.add(unit)
        timestamps.append(sample["host_received_ns"])
    aligned = len(units) == 1 and bool(timestamps) and (max(timestamps)-min(timestamps))/1e9 <= 2
    return {
        "metric": metric_key, "measured": nonnegative_number(total) if counted and aligned else None,
        "peak": nonnegative_number(peak) if counted and aligned and peaks_complete else None,
        "attainment_pct": attain(total, peak) if counted and aligned and peaks_complete else None,
        "status": "COMPLETE" if counted == len(inventory) and aligned and counted else "PARTIAL",
        "counted_gpus": counted, "total_gpus": len(inventory),
    }
