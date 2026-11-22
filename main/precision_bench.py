#!/usr/bin/env python3
"""GPU Pulse precision sweep.

Executes real CUDA kernels per available precision. Unsupported formats remain
BACKEND_REQUIRED; failed launches remain EXECUTION_FAILED. This is an active
benchmark, intentionally separate from the passive CUPTI observer.
"""
from __future__ import annotations

import argparse
import json
import math

VECTOR = ("INT4", "INT8", "INT16", "INT32", "FP16", "BF16", "FP32", "FP64")
MATRIX = ("INT1", "INT2", "INT4", "INT8", "MXFP4", "NVFP4", "MXFP6",
          "FP8", "MXFP8", "FP16", "BF16", "FP32", "TF32", "FP64")
NEEDS_BACKEND = {
    ("Vector", "INT4"): "Native packed INT4 pointwise kernel required",
    ("Matrix", "INT1"): "Binary packed MMA backend required",
    ("Matrix", "INT2"): "Packed INT2 MMA backend required",
    ("Matrix", "INT4"): "Packed INT4 MMA backend required",
    ("Matrix", "MXFP4"): "Native block-scaled MXFP4 backend required",
    ("Matrix", "NVFP4"): "Native NVFP4 backend with per-block scaling required",
    ("Matrix", "MXFP6"): "Native block-scaled MXFP6 backend required",
    ("Matrix", "MXFP8"): "Native block-scaled MXFP8 backend required",
}


def supported_formats():
    """Return the full catalog, not a claim of hardware support."""
    return {"Vector": list(VECTOR), "Matrix": list(MATRIX)}


def status_row(family, precision, status, reason="", **kwargs):
    return dict(family=family, precision=precision, status=status,
                reason=reason, **kwargs)


def _time_cuda(torch, work, operations, divisor, unit, repeats):
    for _ in range(3):
        work()
    torch.cuda.synchronize()
    a = torch.cuda.Event(enable_timing=True)
    b = torch.cuda.Event(enable_timing=True)
    seconds = []
    for _ in range(repeats):
        a.record()
        work()
        b.record()
        b.synchronize()
        value = a.elapsed_time(b) / 1000.0
        if value > 0 and math.isfinite(value):
            seconds.append(value)
    if not seconds:
        raise RuntimeError("CUDA event timing did not produce a positive duration")
    elapsed = min(seconds)
    return dict(value=operations / elapsed / divisor, unit=unit,
                operations=int(operations), kernel_seconds=elapsed)


def _vector(torch, precision, device, elements, repeats):
    dtypes = {
        "INT8": torch.int8, "INT16": torch.int16, "INT32": torch.int32,
        "FP16": torch.float16, "BF16": torch.bfloat16,
        "FP32": torch.float32, "FP64": torch.float64,
    }
    dtype = dtypes[precision]
    if dtype.is_floating_point:
        a = torch.randn(elements, device=device, dtype=dtype)
        b = torch.randn_like(a)
    else:
        a = torch.randint(0, 8, (elements,), device=device, dtype=dtype)
        b = torch.randint(0, 8, (elements,), device=device, dtype=dtype)
    out = torch.empty_like(a)
    floating = dtype.is_floating_point
    return dict(
        _time_cuda(torch, lambda: torch.add(a, b, out=out), elements,
                   1e9, "GFLOP/s" if floating else "GOP/s", repeats),
        backend="PyTorch ATen CUDA pointwise add",
    )


def _matrix(torch, precision, device, size, repeats):
    dtypes = {"FP16": torch.float16, "BF16": torch.bfloat16,
              "FP32": torch.float32, "FP64": torch.float64}
    operations = 2 * size**3
    if precision == "INT8":
        if not hasattr(torch, "_int_mm"):
            raise RuntimeError("This PyTorch build does not expose native CUDA _int_mm")
        a = torch.randint(-4, 4, (size, size), dtype=torch.int8, device=device)
        b = torch.randint(-4, 4, (size, size), dtype=torch.int8, device=device)
        return dict(
            _time_cuda(torch, lambda: torch._int_mm(a, b), operations,
                       1e9, "GOP/s", repeats),
            backend="PyTorch CUDA INT8 GEMM, INT32 accumulation",
        )
    if precision == "FP8":
        if torch.cuda.get_device_capability(device)[0] < 9:
            raise RuntimeError("Native FP8 GEMM needs a compatible GPU backend (typically SM90+)")
        if not hasattr(torch, "_scaled_mm") or not hasattr(torch, "float8_e4m3fn"):
            raise RuntimeError("Native torch._scaled_mm / float8_e4m3fn unavailable")
        a = (torch.randn(size, size, device=device) * 0.1).to(torch.float8_e4m3fn)
        b = (torch.randn(size, size, device=device) * 0.1).to(torch.float8_e4m3fn)
        scale = torch.tensor(1.0, dtype=torch.float32, device=device)
        def work():
            try:
                return torch._scaled_mm(a, b, scale_a=scale, scale_b=scale)
            except TypeError:
                return torch._scaled_mm(a, b, scale, scale)
        return dict(
            _time_cuda(torch, work, operations, 1e12, "TFLOP/s", repeats),
            backend="PyTorch CUDA scaled FP8 GEMM",
        )
    dtype = dtypes.get(precision, torch.float32)
    a = torch.randn(size, size, dtype=dtype, device=device)
    b = torch.randn_like(a)
    out = torch.empty_like(a)
    old = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = precision == "TF32"
        measured = _time_cuda(
            torch, lambda: torch.mm(a, b, out=out),
            operations, 1e12, "TFLOP/s", repeats,
        )
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old
    # TF32 policy does not prove the selected cuBLAS kernel used Tensor Cores.
    backend = ("PyTorch/cuBLAS FP32 storage (TF32 allowed; MMA unverified)"
               if precision == "TF32" else "PyTorch/cuBLAS GEMM")
    return dict(measured, backend=backend)


def sweep(size=512, vector_elements=1 << 20, repeats=8, device_index=0):
    """Active workload sweep; DO NOT run alongside passive observer by default."""
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    device = torch.device("cuda", device_index)
    torch.cuda.set_device(device)
    torch.empty(1, device=device)
    torch.cuda.synchronize()
    rows = []
    for family, formats in (("Vector", VECTOR), ("Matrix", MATRIX)):
        for precision in formats:
            key = (family, precision)
            if key in NEEDS_BACKEND:
                rows.append(status_row(family, precision, "BACKEND_REQUIRED",
                                       NEEDS_BACKEND[key]))
                continue
            try:
                if family == "Vector":
                    measured = _vector(torch, precision, device, vector_elements, repeats)
                else:
                    measured = _matrix(torch, precision, device, size, repeats)
                rows.append(status_row(family, precision, "MEASURED", **measured))
            except (RuntimeError, TypeError, ValueError, AttributeError) as exc:
                rows.append(status_row(family, precision, "EXECUTION_FAILED", str(exc)[:300]))
    return {"gpu": torch.cuda.get_device_name(device_index),
            "compute_capability": list(torch.cuda.get_device_capability(device_index)),
            "note": "GEMM effective throughput or vector pointwise throughput; NOT total hardware FLOP/s",
            "results": rows}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--vector-elements", type=int, default=1 << 20)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--output", default="precision_results.json")
    args = parser.parse_args(argv)
    if args.size <= 0 or args.vector_elements <= 0 or args.repeats <= 0:
        parser.error("size, vector-elements and repeats must be positive")
    result = sweep(args.size, args.vector_elements, args.repeats, args.device)
    with open(args.output, "w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)
    for row in result["results"]:
        val = f'{row["value"]:.3f} {row["unit"]}' if row["status"] == "MEASURED" else row.get("reason", "")
        print(f'{row["family"]:7} {row["precision"]:6} {row["status"]:18} {val}')
    print(f'Saved {args.output}')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
