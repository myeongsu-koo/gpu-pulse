# GPU Throughput Observatory — CUPTI PM Sampling prototype

**Purpose:** Collect hardware-measured TFLOP/s, DRAM GB/s and L1/L2 cache hit rate from the active GPU without converting DCGM utilization percentages into throughput. Ship live data to a built-in read-only local dashboard and NDJSON archive.

**Status:** Source and CPU-side unit tests provided; NOT validated against physical L40S, L4 or A100 in this environment. The first action on your GPU must be `--probe`. A GPU may expose PM Sampling without exposing any FLOP counters compatible with the chosen single pass; the UI displays N/A rather than invented throughput.

## Requirements

- Linux x86_64 / aarch64 with supported NVIDIA driver, CUDA Toolkit and CUPTI libraries (`libcupti`, `nvperf*` accessible).
- CUPTI Python >=13.2; permissions to access NVIDIA GPU performance counters; supported A100/L4/L40S.
- A GPU workload may run concurrently, but whether counters reflect other processes is not guaranteed by this code. **Do not interpret measurements as per-Pod, per-process or per-model**.
- Do not simultaneously run another profiler using the same GPU performance monitors. If DCGM profiling fields conflict, disable those DCGM groups *for the test*, keeping normal DCGM health monitoring.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# MUST run first on each physical GPU. It lists candidate counters and compatible sets.
python observer.py --device 0 --probe > gpu0_probe.json

# Run background PM sampling; no synthetic CUDA workload is launched.
python observer.py --device 0 --web --output gpu0.jsonl \
  --poll-ms 500 --interval-ms 10

# A100 (GA100): uses SYSCLK cycles, NOT 10ms wall-clock time.
python observer.py --device 0 --web --sysclk-cycles 10000000

# SSH port forwarding for remote server:
ssh -L 8765:127.0.0.1:8765 gpu-server
# open http://127.0.0.1:8765 in local browser
```

Metrics chosen automatically from ONE CUPTI *single-pass set*. To force one explicitly:

```bash
python observer.py --probe
python observer.py --metric-set 'EXACT_NAME_FROM_PROBE' --web
python observer.py --require dram_read,dram_write --web
```

**`--require` uses internal metric keys:** `tensor_fp16`, `tensor_bf16`, `tensor_tf32`, `tensor_fp8`, `fadd`, `fmul`, `ffma`, `dram_read`, `dram_write`, `l1_hit`, `l2_hit`.

## Expected output

`gpu0.jsonl` contains one JSON object per CUPTI sample, with `gpu_start_ns`, `gpu_end_ns`, `sample_duration_ms`, `raw_metrics`, `measurements` and coverage. A typical structural example (not measured data):

```json
{"gpu_start_ns": 1, "gpu_end_ns": 2, "measurements": {"tensor_tflops": null, "cuda_fp32_tflops": null, "dram_read_gbps": null, "dram_write_gbps": null, "l1_hit_pct": null, "l2_hit_pct": null}}
```

The UI shows N/A for unavailable values. The `--probe` output establishes candidate *base metric presence*. Actual metric sub-metrics and hardware scheduling are validated by `Collector.configure()`. If configuration fails for a combination, use a different set/metric combination on that GPU; never silently switch to activity-based estimates.

## Metric provenance and limits

- Tensor TFLOP/s: sum of *only supported, named* FP16, BF16, TF32, FP8 Tensor-path **math-operations-per-second** metrics, divided by 1e12. It excludes unmapped precisions and INT8 integer operations. This number may not represent **total** Tensor throughput.
- FP32 TFLOP/s: `FADD + FMUL + 2 × FFMA` thread-instruction rates / 1e12, only if **all three** are actually collectable in one pass. SASS counters are typically not PM-sampling compatible; expect N/A rather than a guess.
- DRAM GB/s: actual bytes read/written per second / 1e9; A100 device DRAM is HBM, L4/L40S is GDDR6.
- L1/L2 hit %: PerfWorks sector-based hit percentages when collectable. L1 bypass and sector granularity mean cache hit is not equal to cache benefit.
- Computed TFLOP/s and GB/s are distinct hardware measurement units, not utilization percentages. `*.sum.per_second` has ALREADY normalized the counter by time; **do not divide it by the sample duration again**.
- Different GPU metric sets cannot be fused into a simultaneous six-metric measurement by alternating collection windows; the application purposely never does this.
- PM sampling has nonzero overhead and risks lost samples; validate overhead against actual QPS, TTFT and TPOT.
- A100 uses GPU SYSCLK clock-count sampling; L4/L40S use time-based sampling.

## Tests (no GPU required)

```bash
python -m unittest discover -s tests -v
python -m py_compile observer.py metrics.py webui.py
```

## Official NVIDIA references

- [CUPTI PM Sampling and single-pass restrictions](https://docs.nvidia.com/cupti/main/main.html#cupti-pm-sampling-api)
- [CUPTI Python Collector API](https://docs.nvidia.com/cupti-python/api-reference/topics/pm_sampling.html)
- [CUPTI Python ProfilerHost metric-set query API](https://docs.nvidia.com/cupti-python/api-reference/topics/profiler_host.html)
- [CUPTI Python tutorial](https://docs.nvidia.com/cupti-python/tutorials/tutorials.html)
- [NVIDIA developer-tools permission requirement](https://developer.nvidia.com/ERR_NVGPUCTRPERM)
