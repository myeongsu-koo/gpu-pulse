"""SKU-safe peak matching: never substitute matrix throughput for CUDA FLOP/s."""
from __future__ import annotations
import json
from pathlib import Path

REGISTRY_PATH = Path(__file__).with_name("peak_specs.json")


def load_registry(path=None):
    with open(path or REGISTRY_PATH, encoding="utf-8") as f:
        obj = json.load(f)
    if obj.get("schema_version") != 1:
        raise ValueError("Unsupported peak registry schema")
    return obj["skus"]


def _normalized(name):
    return " ".join(name.upper().replace("NVIDIA", "").split())


def match_peak(gpu, registry=None):
    """Return explicit match or status; ambiguous GPU identifiers do not imply SXM."""
    registry = load_registry() if registry is None else registry
    name = _normalized(gpu.get("model", ""))
    memory = gpu.get("memory_mib")
    candidates = []
    for spec in registry:
        aliases = spec["aliases"]
        if not any((" " + _normalized(alias) + " ") in (" " + name + " ") for alias in aliases):
            continue
        if spec.get("requires") and not all(word in name for word in spec["requires"]):
            continue
        if spec.get("memory_gb") and memory is not None:
            # nvidia-smi MiB, tolerate display rounding / reserved memory.
            if abs(memory / 1024 - spec["memory_gb"]) > 3:
                continue
        candidates.append(spec)
    if len(candidates) == 1:
        spec = candidates[0]
        return {"status": "MATCHED", "sku": spec["sku"], "spec": spec}
    return {"status": "SKU_AMBIGUOUS" if candidates else "SPEC_UNAVAILABLE",
            "sku": None, "candidates": [s["sku"] for s in candidates]}


def attain(measured, peak):
    if not isinstance(measured, (int, float)) or isinstance(measured, bool):
        return None
    if not isinstance(peak, (int, float)) or isinstance(peak, bool):
        return None
    import math
    if not all(map(math.isfinite, (measured, peak))) or measured < 0 or peak <= 0:
        return None
    return 100.0 * measured / peak


def peak_for(spec, metric):
    """metric keys carry execution path and precision, e.g. fp16.tensor.dense."""
    if not spec:
        return None
    return spec.get("compute_peaks", {}).get(metric)


def measured_compute(precision_measurements, spec):
    """Only calculate attained percent for *actually measured* matching precision."""
    result = {}
    for metric, record in precision_measurements.items():
        peak = peak_for(spec, metric)
        value = record.get("value") if isinstance(record, dict) else None
        result[metric] = {"measured": value,
                          "peak": peak["value"] if peak else None,
                          "unit": peak["unit"] if peak else record.get("unit") if isinstance(record, dict) else None,
                          "attainment_pct": attain(value, peak["value"]) if peak else None}
    return result
