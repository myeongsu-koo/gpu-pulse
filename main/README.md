# GPU Pulse operator guide

GPU Pulse streams CUPTI measurements to a read-only local dashboard and writes NDJSON samples. Choose the standalone collector for a specific CUDA ordinal or the node agent to discover physical GPUs and run per-GPU workers.

## Requirements

- Linux x86_64 / aarch64 with supported NVIDIA driver, CUDA Toolkit and CUPTI libraries (`libcupti`, `nvperf*` accessible).
- CUPTI Python >=13.2 and permissions to access NVIDIA GPU performance counters. Inventory uses the CUDA Driver API. GPU Pulse does not query NVML.
- A GPU workload may run concurrently, but whether counters reflect other processes is not guaranteed by this code. **Do not interpret measurements as per-Pod, per-process or per-model**.
- Do not simultaneously run another profiler using the same GPU performance monitors. If DCGM profiling fields conflict, disable those DCGM groups *for the test*, keeping normal DCGM health monitoring.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Inspect candidate counters and compatible sets on the selected device.
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

## Multi-GPU mode

```bash
python node_agent.py --output-dir ./samples --interval-ms 100 --poll-ms 500
ssh -L 8766:127.0.0.1:8766 gpu-server
```

| View or API | Route |
| --- | --- |
| Node Summary | `/` or `/summary` on port 8766 |
| GPU Detail from Node Summary | `/gpu/<uuid>` on port 8766 |
| Standalone GPU Detail | `/` or `/dashboard.html` on port 8765 |
| Standalone samples and collector state | `/api/state` |
| Node aggregate and inventory | `/api/v1/node/summary` and `/api/v1/gpus` |
| Per-GPU measurements and history | `/api/v1/gpus/<uuid>/metrics` and `/api/v1/gpus/<uuid>/history` |

The agent enumerates CUDA-visible devices and uses their CUDA ordinals directly. Devices whose CUDA names identify them as MIG instances appear with collection disabled. Each worker streams samples into a per-GPU in-memory ring buffer limited by `--history` (default 256 samples). The sample window resets when the agent restarts. The output directory contains worker stderr logs. Exit code 0 appears as `STOPPED`. Nonzero exits and launch errors appear as `FAILED`. Node aggregation aligns samples on host receipt time and reports partial coverage when measurements are missing. Cache hit rates are available per GPU.

## Generate an offline HTML report

Run the exporter from this checkout with Python 3. It reads local observer NDJSON, standalone API captures, or a node GPU’s recent-history JSON. The resulting HTML embeds its data, styles, and scripts. Open it directly in a browser on an isolated network.

```bash
# Existing standalone capture
python exporter_html.py gpu0.jsonl -o gpu0-report.html --model 'NVIDIA H200'

# Save the retained GPU samples from a node agent while it is running.
# Substitute the UUID shown in Node Summary.
curl -fsS http://127.0.0.1:8766/api/v1/gpus/GPU-UUID/history -o gpu0-history.json
python exporter_html.py gpu0-history.json -o gpu0-report.html
```

The `curl` command is only for copying a capture from a running agent. Report generation and viewing work from the saved file with Python and a browser, without GPU libraries or network access.

## Peak lookup and memory units

- Hyphenated A100 names such as `NVIDIA A100-SXM4-40GB` and `NVIDIA A100-PCIE-80GB` resolve to their specific form factors. A bare A100 label leaves the form factor unresolved.
- Driver names `NVIDIA H100 80GB HBM3` and `NVIDIA H200` resolve to SXM references. H100 PCIe and H100/H200 NVL remain outside the current registry.
- The matcher accepts reported **total memory in MiB**, with per-SKU ranges for L40S and B300. Example totals include L40S at 46,068 MiB and B300 at 275,040 MiB. In Linux GPU Burn’s initialization line, the first memory number is total memory in MiB. The parenthesized numbers describe available memory and test allocation.

Driver names and peak references come from the [NVIDIA driver product list](https://download.nvidia.com/solaris/580.142/README/supportedchips.html) and [A100 datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf). Example totals and units come from the [Cisco L40S reference deployment](https://www.cisco.com/c/en/us/td/docs/unified_computing/ucs/UCS_CVDs/flexpod_rag_nim.pdf), [Oracle B300 quickstart](https://github.com/oracle-quickstart/oci-gpu-quickstarts/blob/main/nvidia/B300/README-B300.md), and [GPU Burn source](https://github.com/wilicc/gpu-burn/blob/master/gpu_burn-drv.cpp).

## Standalone sample output

When you pass `--output gpu0.jsonl` to the standalone observer, it appends one JSON object per CUPTI sample, with `gpu_start_ns`, `gpu_end_ns`, `sample_duration_ms`, `raw_metrics`, `measurements` and coverage. A typical structural example (not measured data):

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
python -m py_compile *.py
```

## Official NVIDIA references

- [CUPTI PM Sampling and single-pass restrictions](https://docs.nvidia.com/cupti/main/main.html#cupti-pm-sampling-api)
- [CUPTI Python Collector API](https://docs.nvidia.com/cupti-python/api-reference/topics/pm_sampling.html)
- [CUPTI Python ProfilerHost metric-set query API](https://docs.nvidia.com/cupti-python/api-reference/topics/profiler_host.html)
- [CUPTI Python tutorial](https://docs.nvidia.com/cupti-python/tutorials/tutorials.html)
- [NVIDIA developer-tools permission requirement](https://developer.nvidia.com/ERR_NVGPUCTRPERM)
