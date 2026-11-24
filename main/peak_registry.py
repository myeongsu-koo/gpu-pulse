"""SKU-safe peak matching: never substitute matrix throughput for CUDA FLOP/s."""
from __future__ import annotations
import json
import re
from pathlib import Path

REGISTRY_PATH = Path(__file__).with_name("peak_specs.json")


def load_registry(path=None):
    with open(path or REGISTRY_PATH, encoding="utf-8") as f:
        obj = json.load(f)
    if obj.get("schema_version") != 1:
        raise ValueError("Unsupported peak registry schema")
    return obj["skus"]


def _normalized(name):
    tokens = re.findall(r"[A-Z0-9]+", name.upper())
    return " ".join(tokens[1:] if tokens[:1] == ["NVIDIA"] else tokens)


def match_peak(gpu, registry=None):
    """Match driver names or explicit variant markers and validate total memory in MiB."""
    registry = load_registry() if registry is None else registry
    name = _normalized(gpu.get("model") or "")
    tokens = set(name.split())
    if any(re.fullmatch(r"SXM[0-9]+", token) for token in tokens):
        tokens.add("SXM")
    if "SXM" in tokens and tokens.intersection({"PCIE", "NVL"}):
        return {"status": "SPEC_UNAVAILABLE", "sku": None, "candidates": []}
    memory = gpu.get("memory_mib")
    capacity = re.search(r"\b(\d+) ?GB\b", name)
    candidates = []
    for spec in registry:
        exact = any(name == _normalized(alias) for alias in spec.get("exact_names", ()))
        if not exact:
            if not any(f" {_normalized(alias)} " in f" {name} " for alias in spec["aliases"]):
                continue
            if not tokens.issuperset(spec.get("requires", ())):
                continue
        nominal = spec.get("memory_gb")
        if capacity and "A100" in tokens and int(capacity.group(1)) != nominal:
            continue
        if nominal and memory is not None:
            # Per-SKU ranges account for the reported capacity after reservations.
            lower, upper = spec.get("memory_mib_range", ((nominal - 3) * 1024, (nominal + 3) * 1024))
            if not lower <= memory <= upper:
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
        verified = not isinstance(record, dict) or record.get("counter_unit_verified") is not False
        result[metric] = {"measured": value,
                          "counter_unit_verified": verified,
                          "peak": peak["value"] if peak else None,
                          "unit": peak["unit"] if peak else record.get("unit") if isinstance(record, dict) else None,
                          "attainment_pct": attain(value, peak["value"]) if peak and verified else None}
    return result
