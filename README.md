# GPU Pulse

**GPU execution & throughput observability powered by NVIDIA CUPTI**

GPU Pulse is an experimental GPU-performance observability project that combines **hardware performance counters** with **measured GEMM throughput**. Its Colab proof of concept profiles a 60-second variable workload and visualizes the results on a synchronized timeline.

## Dashboard preview

> GPU Pulse metric cards
<div align="left">
  <table>
    <tr>
      <td align="center" width="380">
        <img src="https://github.com/user-attachments/assets/61a273a9-c189-4257-a284-b776b586846e" alt="image" width="100%" />
        <sub>NVIDIA L4 GPU Pulse Dashboard (AD104)</sub>
      </td>
    </tr>
  </table>
</div>

> Measured GPU throughput & hardware timeline
<div align="left">
  <table>
    <tr>
      <td align="center" width="380">
        <img width="760" alt="image" src="https://github.com/user-attachments/assets/8ab2ff38-c9ab-45df-ad3d-d163040e254e" />
        <img width="760" alt="image" src="https://github.com/user-attachments/assets/4380c353-6abe-471b-9a92-fb83a2554fe3" />
        <sub>NVIDIA L4 GPU Pulse Dashboard (AD104)</sub>
      </td>
    </tr>
  </table>
</div>

*Captured on an NVIDIA L4 (AD104) with CUPTI Python 13.5.0. The screenshots show a real test run, not synthetic dashboard data.*

## What GPU Pulse measures

| Metric | Source | Interpretation |
| --- | --- | --- |
| FP16 GEMM throughput | CUDA Events + known GEMM operations | Effective workload TFLOP/s during active GEMM execution |
| FP32 GEMM throughput | CUDA Events + known GEMM operations | Effective workload TFLOP/s during active GEMM execution |
| DRAM read/write | CUPTI PM Sampling | Hardware counter-derived throughput, GB/s |
| L1/L2 cache hit | CUPTI PM Sampling | Cache hit metrics, subject to metric-specific semantics |
| GR/SM elapsed cycles | CUPTI PM Sampling | Raw cycle counter measurements |
| FP32 hardware FLOP/s | CUPTI, if compatible counters are available | Not available in the demonstrated L4 configuration |
| Tensor pipeline activity | CUPTI, if a compatible metric is available | Not available in the demonstrated L4 configuration |

**Effective GEMM TFLOP/s is not equivalent to a GPU-wide hardware FLOP counter.** GEMM throughput is calculated from the work performed by a known benchmark. Counters are reported separately; unavailable hardware metrics appear as **N/A**, not as zeros or estimated values.

## Variable workload demo

The 60-second demonstration runs a changing GPU workload, enabling comparison between compute and memory phases:

| Interval | Workload |
| --- | --- |
| 0–8 s | Idle |
| 8–20 s | FP16 GEMM, increasing load |
| 20–32 s | FP32 GEMM, increasing load |
| 32–42 s | Memory-intensive operations |
| 42–54 s | Mixed compute and memory |
| 54–60 s | Idle |

The dashboard aligns observations on one-second reporting windows. The FP16 and FP32 benchmark operations are scheduled in phases rather than being guaranteed to execute at the exact same instant.

## Validated environment

- **GPU:** NVIDIA L4, AD104 (compute capability 8.9)
- **CUPTI Python:** 13.5.0
- **CUPTI C package:** 13.5.25
- **Available single-pass metric sets:** `TriageSCG`, `TriageAC`
- **Collected:** GR/SM cycles, DRAM read/write, L1/L2 cache metrics
- **Observed sample count:** 59 in a ~58-second valid capture
- **Additional FP32/Tensor counters:** Not compatible with the selected single-pass collection set in this run

**L40S, A100, H100/H200 and Blackwell-series support must be verified per GPU and CUPTI version.** Automatic detection is not evidence that every metric is supported on every architecture.

## Getting started

The current implementation is a Google Colab proof of concept. It requires a supported NVIDIA GPU runtime and a CUPTI installation with working PM Sampling APIs.

1. Open the GPU Pulse Colab notebook.
2. Select a GPU-enabled runtime.
3. Run the collection and variable-workload demo.
4. Inspect the measured FLOP/s and hardware timeline.

> **Status:** Experimental. This project currently demonstrates post-capture visualization, not a continuously streaming production monitoring agent. A standalone collector and CLI are planned.

## References

- [NVIDIA CUPTI](https://docs.nvidia.com/cupti/)
- [CUPTI Python PM Sampling](https://docs.nvidia.com/cupti-python/api-reference/topics/pm_sampling.html)
- [CUDA Event API](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html)
- [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)
