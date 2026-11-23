# GPU Pulse

GPU Pulse collects live NVIDIA GPU performance counters with CUPTI PM Sampling. Use Node Summary to scan a multi-GPU host and open any GPU’s Detail page. The standalone collector serves the same GPU Detail dashboard for a single device.

## Run on a GPU host

```bash
cd main
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Check the counter sets exposed by a CUDA device
python observer.py --device 0 --probe

# Multi-GPU Node Summary at http://127.0.0.1:8766/
python node_agent.py --output-dir ./samples

# Or open standalone GPU Detail at http://127.0.0.1:8765/
python observer.py --device 0 --web --output gpu0.jsonl
```

Both collectors run passively. The node agent starts a worker per eligible GPU and writes each worker’s sample stream and stderr log to the output directory. See the [operator guide](main/README.md) for requirements, routes, and remote access.

| View | What it shows |
| --- | --- |
| Node Summary | Physical GPU inventory, collector status, node DRAM and compute aggregates, and measurement coverage |
| GPU Detail | Live compute, DRAM and cache charts, measured versus dense peak by precision, temperature, power, clocks, VRAM, and metric provenance |

The registry covers T4, L4, L40S, A100 40/80 GB in PCIe and SXM, H100 SXM, H200 SXM, B200, and B300. A registry match supplies theoretical peaks. Live counter availability depends on the GPU, CUPTI version, and selected single-pass set. The dashboard displays missing values as **N/A** and flags partial node coverage. Node time alignment uses host receipt time.

## Colab demo

The [Colab notebook](main/notebooks/GPU_Pulse_All_Precisions.ipynb) runs an active variable workload and visualizes timed GEMM throughput alongside hardware counters. Timed GEMM throughput measures benchmark work divided by execution time. CUPTI metrics describe sampled hardware activity.

<img src="https://github.com/user-attachments/assets/61a273a9-c189-4257-a284-b776b586846e" alt="GPU Pulse Colab dashboard captured on NVIDIA L4" width="650">

The preview is from an NVIDIA L4 capture with CUPTI Python 13.5.0.

## References

- [NVIDIA CUPTI PM Sampling](https://docs.nvidia.com/cupti/main/main.html#cupti-pm-sampling-api)
- [CUPTI Python PM Sampling](https://docs.nvidia.com/cupti-python/api-reference/topics/pm_sampling.html)
- [NVIDIA System Management Interface](https://docs.nvidia.com/deploy/nvidia-smi/index.html)
