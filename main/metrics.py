"""Metric planning and unit-safe transformation.  No GPU/CUPTI imports.

IMPORTANT: A metric is considered collectable only when the current GPU lists
its base metric in an actual CUPTI single-pass set; configure() is the final
source of truth. We deliberately do NOT estimate FLOP/s from utilization.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable
import math


@dataclass(frozen=True)
class MetricSpec:
    key: str             # stable internal key
    base: str            # PerfWorks base metric
    suffix: str          # evaluator sub-metric; no implicit conversions
    category: str        # tensor|fp32|dram|cache
    factor: float = 1.0 # FP32 FFMA = two arithmetic operations
    weight: float = 1.0

    @property
    def name(self) -> str:
        return self.base + self.suffix


SPECS = [
    # IMPORTANT: precision paths are disjoint; integer tensor OPS are NOT FLOPs.
    MetricSpec("tensor_fp16", "sm__ops_path_tensor_src_fp16_dst_fp32", ".sum.per_second", "tensor", weight=4),
    MetricSpec("tensor_bf16", "sm__ops_path_tensor_src_bf16_dst_fp32", ".sum.per_second", "tensor", weight=4),
    MetricSpec("tensor_tf32", "sm__ops_path_tensor_src_tf32_dst_fp32", ".sum.per_second", "tensor", weight=4),
    MetricSpec("tensor_fp8", "sm__ops_path_tensor_src_fp8_dst_fp32", ".sum.per_second", "tensor", weight=3),
    # SASS counters are often not available for periodic PM sampling; NEVER
    # substitute FMA pipeline utilization or issue rate for real FLOPs.
    MetricSpec("fadd", "smsp__sass_thread_inst_executed_op_fadd_pred_on", ".sum.per_second", "fp32", 1, 3),
    MetricSpec("fmul", "smsp__sass_thread_inst_executed_op_fmul_pred_on", ".sum.per_second", "fp32", 1, 3),
    MetricSpec("ffma", "smsp__sass_thread_inst_executed_op_ffma_pred_on", ".sum.per_second", "fp32", 2, 4),
    # L4/AD104 validated alternative: 32-byte sectors per sampling interval.
    MetricSpec("dram_read_sectors", "dramc__sectors_op_read", ".sum", "dram_sectors", weight=7),
    MetricSpec("dram_write_sectors", "dramc__sectors_op_write", ".sum", "dram_sectors", weight=7),
    MetricSpec("dram_read", "dram__bytes_read", ".sum.per_second", "dram", weight=8),
    MetricSpec("dram_write", "dram__bytes_write", ".sum.per_second", "dram", weight=8),
    MetricSpec("l1_hit", "l1tex__t_sector_hit_rate", ".pct", "cache", weight=2),
    MetricSpec("l2_hit", "lts__t_sector_hit_rate", ".pct", "cache", weight=3),
    MetricSpec("l2_hit_realtime", "lts__average_t_sector_hit_rate_realtime", ".pct", "cache", weight=3),
]

SECTOR_BYTES = 32  # NVIDIA memory sector size (bytes)

KEY_TO_SPEC = {s.key: s for s in SPECS}
TARGETS = ("tensor_tflops", "cuda_fp32_tflops", "dram_read_gbps", "dram_write_gbps", "l1_hit_pct", "l2_hit_pct")


def base_name(metric: str) -> str:
    return metric.split(".", 1)[0]


def select_for_set(set_metrics: Iterable[str]) -> list[MetricSpec]:
    bases = {base_name(m) for m in set_metrics}
    return [s for s in SPECS if s.base in bases]


def score_set(specs: list[MetricSpec]) -> float:
    keys = {s.key for s in specs}
    score = sum(s.weight for s in specs)
    # Do not award "FP32 TFLOP/s" if ADD/MUL/FMA are incomplete.
    if {"fadd", "fmul", "ffma"}.issubset(keys):
        score += 10
    if ({"dram_read", "dram_read_sectors"} & keys and
            {"dram_write", "dram_write_sectors"} & keys):
        score += 6
    return score


def choose_set(sets_to_metrics: dict[str, list[str]], required: Iterable[str] = ()) -> tuple[str, list[MetricSpec]]:
    required = set(required)
    choices = []
    for name, available in sets_to_metrics.items():
        selected = select_for_set(available)
        if not required.issubset({s.key for s in selected}):
            continue
        if selected:
            choices.append((score_set(selected), len(selected), name, selected))
    if not choices:
        raise ValueError("No suitable single-pass metric set. Run --probe to inspect chip capabilities.")
    choices.sort(key=lambda x: (-x[0], -x[1], x[2]))
    _, _, name, selected = choices[0]
    return name, selected


def _safe_num(x: object) -> float | None:
    try:
        v = float(x)
        return v if math.isfinite(v) and v >= 0 else None
    except (ValueError, TypeError):
        return None


def to_dashboard_values(values: dict[str, object], selected: list[MetricSpec], sample_seconds: float | None = None) -> dict[str, float | None]:
    """Normalize supported counters to TFLOP/s, decimal GB/s and percent.

    Byte .sum.per_second metrics are already rates. L4-style sector .sum metrics
    are per-sample counts and require a valid sample duration in seconds.
    A missing/invalid duration MUST NOT produce an estimated sector rate.
    """
    out: dict[str, float | None] = {k: None for k in TARGETS}
    present = {
        s.key: s for s in selected
        if s.name in values and _safe_num(values[s.name]) is not None
    }
    tensor = [s for s in selected if s.category == "tensor" and s.key in present]
    if tensor:
        out["tensor_tflops"] = sum(float(values[s.name]) * s.factor for s in tensor) / 1e12
    if {"fadd", "fmul", "ffma"}.issubset(present):
        out["cuda_fp32_tflops"] = sum(
            float(values[present[k].name]) * present[k].factor
            for k in ("fadd", "fmul", "ffma")
        ) / 1e12

    duration = _safe_num(sample_seconds)
    # A zero-duration sample is invalid, even though a zero counter is valid.
    duration = duration if duration is not None and duration > 0 else None
    for direct, fallback, target in (
        ("dram_read", "dram_read_sectors", "dram_read_gbps"),
        ("dram_write", "dram_write_sectors", "dram_write_gbps"),
    ):
        if direct in present:
            # Already bytes/second: never divide by the sample duration again.
            out[target] = float(values[present[direct].name]) / 1e9
        elif fallback in present and duration is not None:
            sectors = float(values[present[fallback].name])
            out[target] = sectors * SECTOR_BYTES / duration / 1e9

    if "l1_hit" in present:
        out["l1_hit_pct"] = float(values[present["l1_hit"].name])
    for key in ("l2_hit", "l2_hit_realtime"):
        if key in present:
            out["l2_hit_pct"] = float(values[present[key].name])
            break
    return out


def coverage(selected: list[MetricSpec]) -> dict[str, object]:
    keys = {s.key for s in selected}
    return {
        "tensor_precisions": [k.removeprefix("tensor_") for k in ("tensor_fp16", "tensor_bf16", "tensor_tf32", "tensor_fp8") if k in keys],
        "cuda_fp32_complete": {"fadd", "fmul", "ffma"}.issubset(keys),
        "dram_read": bool({"dram_read", "dram_read_sectors"} & keys),
        "dram_write": bool({"dram_write", "dram_write_sectors"} & keys),
        "l1_hit": "l1_hit" in keys,
        "l2_hit": bool({"l2_hit", "l2_hit_realtime"} & keys),
        "not_counted": [s.key for s in SPECS if s.key not in keys],
        "warning": "A present metric covers ONLY its named paths; null is not zero. For tensor, unlisted precisions are omitted.",
    }

# Precision/path-specific decoded measurements. The existing aggregate
# tensor_tflops field is retained for backward compatibility.
PRECISION_TARGETS = {
    "tensor_fp16": "fp16.tensor.dense",
    "tensor_bf16": "bf16.tensor.dense",
    "tensor_tf32": "tf32.tensor.dense",
    "tensor_fp8": "fp8.tensor.dense",
}


def precision_dashboard_values(values: dict[str, object], selected: list[MetricSpec]) -> dict[str, dict]:
    """Return ONLY genuine decoded arithmetic-operation rates.

    A dense spec can be used as an *explicit reference*, but PM Sampling does
    not independently prove runtime sparsity. That uncertainty is retained in
    comparison_status and must be visible wherever a percentage is shown.
    Integer operation rates are never inferred from FP tensor counters.
    """
    present = {
        s.key: s for s in selected
        if s.name in values and _safe_num(values[s.name]) is not None
    }
    result = {}
    for key, metric in PRECISION_TARGETS.items():
        spec = present.get(key)
        if spec is None:
            continue
        result[metric] = {
            "value": float(values[spec.name]) * spec.factor / 1e12,
            "unit": "TFLOP/s",
            "source_metric": spec.name,
            "comparison_status": "DENSE_REFERENCE_UNVERIFIED",
            "counter_unit_verified": False,
        }
    if {"fadd", "fmul", "ffma"}.issubset(present):
        fops = sum(
            float(values[present[k].name]) * present[k].factor
            for k in ("fadd", "fmul", "ffma")
        )
        result["fp32.cuda.dense"] = {
            "value": fops / 1e12,
            "unit": "TFLOP/s",
            "source_metric": [present[k].name for k in ("fadd", "fmul", "ffma")],
            "comparison_status": "MATCHED",
        }
    return result
