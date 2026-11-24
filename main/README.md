# GPU Pulse operator guide

GPU Pulse streams CUPTI measurements to a read-only local dashboard and writes NDJSON samples. Choose the standalone collector for a specific CUDA ordinal or the node agent to discover physical GPUs and run per-GPU workers.

## Requirements

The observer retains a CUDA primary context on each sampled GPU. This can consume VRAM even though the observer launches no workload kernels. On GPUs configured for `EXCLUSIVE_PROCESS`, the observer and workload can contend for the single process context. Keep the observer off those production GPUs while an exclusive workload needs them. See the [CUDA primary-context API](https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__PRIMARY__CTX.html) and [CUDA compute modes](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#nvidia-smi).

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

The agent enumerates CUDA-visible devices and uses their CUDA ordinals directly. CUDA devices identified as MIG instances remain visible with collection disabled.

Both live dashboards expire samples by age (`--history-seconds`, default 120). A hard cap of 16,384 samples per GPU bounds memory at high sampling rates. The node agent exposes this cap as `--history`. Live GPU charts use half-second buckets. Node charts use duration-weighted per-GPU DRAM averages within one-second host-receipt buckets and sum the available GPU averages. A partial point shows how many GPUs contributed. Host receipt times approximate alignment across devices.

The log directory retains stderr from the latest two worker attempts per GPU. The agent removes its temporary directory on shutdown. Pass `--output-dir` to retain logs across shutdowns. Exit code 0 appears as `STOPPED`. Failed launches and nonzero exits retry with per-GPU backoff from 2 to 60 seconds. The ring buffers reset when the agent restarts.

Both dashboards listen on IPv4 loopback (`127.0.0.1`). The `--bind` and `--host` compatibility flags accept only that address. Use SSH port forwarding for remote viewing.

## Generate an offline HTML report

Run the exporter from this checkout with Python 3. It reads local observer NDJSON, standalone API captures, or a node GPU’s recent-history JSON. The resulting HTML embeds its data, styles, and scripts. Open it directly in a browser on an isolated network.

```bash
# Standalone capture with embedded GPU inventory
python exporter_html.py gpu0.jsonl -o gpu0-report.html --compact

# Save the retained GPU samples from a node agent while it is running.
# Substitute the UUID shown in Node Summary.
curl -fsS http://127.0.0.1:8766/api/v1/gpus/GPU-UUID/history -o gpu0-history.json
python exporter_html.py gpu0-history.json -o gpu0-report.html
```

The `curl` command is only for copying a capture from a running agent. Report generation and viewing work from the saved file with Python and a browser, without GPU libraries or network access.

Standalone JSONL now carries CUDA model, UUID and total VRAM metadata when available. Older captures can use `--model` and `--memory-mib`. The optional `--compact` flag embeds half-second chart averages and the latest-sample detail.

## Profiler conflicts and hardware validation

When CUPTI reports that another profiler owns the counters, the node marks the device `PROFILER_BUSY` and starts retrying after at least 30 seconds. The standalone observer prints the conflict and exits with status 75. Check for another GPU Pulse worker, Nsight, or DCGM hardware profiling. Coordinate with the monitoring owner before pausing DCGM profiling with `dcgmi profile --pause`, and resume it with `dcgmi profile --resume` when done. See [NVIDIA DCGM profiling](https://docs.nvidia.com/datacenter/dcgm/latest/learn/modules/profiling.html).

To verify contention on a test node, confirm DCGM is actively collecting GPU profiling metrics, then start the observer on the same CUDA-visible GPU. Record whether the observer exits with code 75 and the node agent reports `PROFILER_BUSY`. If authorized to pause profiling on that test node, repeat with DCGM profiling paused and verify that samples arrive. Restore DCGM profiling after the check. Record the GPU, driver, CUPTI and DCGM versions and the original error. Some configurations can collect concurrently, so record that outcome if both tools continue to produce metrics.


Both collectors default to 100 ms on time-triggered GPUs. GA100 continues to use SYSCLK-triggered sampling. Production overhead has yet to be measured. On a representative node, alternate identical workload runs with collection off and on. Record GPU model, driver, CUPTI version, collector interval, application throughput and p95/p99 latency. Compare medians across repeated runs at each setting.

Live detail and node summary use the same freshness threshold of five seconds. Saved reports show captured readings regardless of their age. Peak comparisons require matching units and an eligible reference. Tensor captures require explicit counter calibration metadata, including older captures that predate the flag.

Tensor rates are provisional. On a dedicated GPU, run warmed, dense GEMMs of known dimensions with the benchmark's CUDA-event timing and record CUPTI counters for the matching precision. For aligned steady-state windows, compare the duration-weighted counter rate to `2 × M × N × K × completed GEMMs / elapsed seconds`. Repeat across precisions and architectures and confirm which counter counts operations or FMAs before enabling attainment for it. This repository does not yet contain hardware calibration results.

## Peak lookup and memory units

- Hyphenated A100 names such as `NVIDIA A100-SXM4-40GB` and `NVIDIA A100-PCIE-80GB` resolve to their specific form factors. A bare A100 label leaves the form factor unresolved.
- Driver names `NVIDIA H100 80GB HBM3` and `NVIDIA H200` resolve to SXM references. H100 PCIe and H200 NVL have separate registry entries. H100 NVL remains outside the registry.
- The matcher accepts reported **total memory in MiB**, with per-SKU ranges for L40S and B300. Example totals include L40S at 46,068 MiB and B300 at 275,040 MiB. In Linux GPU Burn’s initialization line, the first memory number is total memory in MiB. The parenthesized numbers describe available memory and test allocation.

Hopper and Blackwell compute records include `value_origin`. Derived records carry the published value, unit, divisor and, when relevant, GPU count and sparsity. HGX FP64 rows with ambiguous CUDA-only provenance are excluded from attainment.

Driver names and peak references come from the [NVIDIA driver product list](https://download.nvidia.com/solaris/580.142/README/supportedchips.html) and [A100 datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf). Example totals and units come from the [Cisco L40S reference deployment](https://www.cisco.com/c/en/us/td/docs/unified_computing/ucs/UCS_CVDs/flexpod_rag_nim.pdf), [Oracle B300 quickstart](https://github.com/oracle-quickstart/oci-gpu-quickstarts/blob/main/nvidia/B300/README-B300.md), and [GPU Burn source](https://github.com/wilicc/gpu-burn/blob/master/gpu_burn-drv.cpp).

## Standalone sample output

File output is off by default. When you pass `--output gpu0.jsonl` to the standalone observer, it appends one JSON object per CUPTI sample, with `gpu_start_ns`, `gpu_end_ns`, `sample_duration_ms`, `raw_metrics`, `measurements` and coverage. A typical structural example (not measured data):

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
