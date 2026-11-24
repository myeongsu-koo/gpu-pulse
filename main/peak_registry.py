"""SKU-safe peak matching: never substitute matrix throughput for CUDA FLOP/s."""
from __future__ import annotations
import json
import re
from pathlib import Path

if __package__:
    from .sample_schema import nonnegative_number
else:
    from sample_schema import nonnegative_number

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
    if memory is not None and not nonnegative_number(memory):
        return {"status": "SPEC_UNAVAILABLE", "sku": None, "candidates": []}
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
    measured = nonnegative_number(measured)
    peak = nonnegative_number(peak)
    if measured is None or not peak:
        return None
    return nonnegative_number(measured / peak * 100)


def memory_peak(spec):
    spec = spec or {}
    if spec.get("memory_peak_quality") == "LOWER_BOUND":
        return None
    return nonnegative_number(spec.get("memory_bandwidth_gbps")) or None


def compare_compute(metric, reading, reference):
    """Apply one unit and provenance policy to detail, export and node totals."""
    reading = reading if isinstance(reading, dict) else {}
    reference = reference or {}
    value = nonnegative_number(reading.get("value"))
    source_unit = reading.get("unit")
    peak_unit = reference.get("unit")
    origin = reference.get("value_origin")
    calibration = reading.get("counter_unit_verified")
    verified = calibration is True or (calibration is None and ".tensor." not in metric)
    peak = None if origin == "AMBIGUOUS_SOURCE" else nonnegative_number(reference.get("value")) or None
    compatible = verified and bool(source_unit) and source_unit == peak_unit
    return {
        "measured": value,
        "measured_unit": source_unit,
        "counter_unit_verified": verified,
        "peak": peak,
        "peak_origin": origin,
        "unit": peak_unit or source_unit,
        "attainment_pct": attain(value, peak) if compatible else None,
    }


def measured_compute(precision_measurements, spec):
    references = (spec or {}).get("compute_peaks", {})
    return {
        key: compare_compute(key, precision_measurements.get(key), references.get(key))
        for key in dict.fromkeys((*precision_measurements, *references))
    }
