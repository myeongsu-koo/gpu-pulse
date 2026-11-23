"""Discover physical NVIDIA GPUs and correlate NVML/nvidia-smi with CUDA ordinals.

No guessed mapping: if CUDA PCI bus lookup fails a GPU is shown but not profiled.
MIG child devices are not treated as independent physical GPUs.
"""
from __future__ import annotations
import csv
import ctypes
import io
import subprocess


def _bus_id(raw):
    # CUDA commonly returns 0000:65:00.0; NVML sometimes 00000000:65:00.0.
    pieces = str(raw).strip().lower().split(":")
    if len(pieces) != 3:
        return str(raw).strip().lower()
    return f"{int(pieces[0], 16):04x}:{pieces[1]}:{pieces[2]}"


def cuda_bus_map():
    """Return {normalized PCI bus ID: CUDA ordinal}, or {} if CUDA is inaccessible."""
    try:
        driver = ctypes.CDLL("libcuda.so.1")
        driver.cuInit.argtypes = [ctypes.c_uint]
        driver.cuInit.restype = ctypes.c_int
        driver.cuDeviceGetCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
        driver.cuDeviceGetCount.restype = ctypes.c_int
        driver.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        driver.cuDeviceGet.restype = ctypes.c_int
        driver.cuDeviceGetPCIBusId.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
        driver.cuDeviceGetPCIBusId.restype = ctypes.c_int
        if driver.cuInit(0):
            return {}
        count = ctypes.c_int()
        if driver.cuDeviceGetCount(ctypes.byref(count)):
            return {}
        mapping = {}
        for ordinal in range(count.value):
            device = ctypes.c_int()
            bus = ctypes.create_string_buffer(64)
            if driver.cuDeviceGet(ctypes.byref(device), ordinal) == 0 and driver.cuDeviceGetPCIBusId(bus, 64, device.value) == 0:
                mapping[_bus_id(bus.value.decode())] = ordinal
        return mapping
    except (OSError, AttributeError, ValueError):
        return {}


def discover_gpus(command=None, cuda_mapping=None):
    """Return GPU inventory with immutable UUIDs; raises if nvidia-smi fails."""
    if command is None:
        command = ["nvidia-smi", "--query-gpu=index,uuid,pci.bus_id,name,memory.total", "--format=csv,noheader,nounits"]
    output = subprocess.run(command, capture_output=True, text=True, check=True, timeout=15).stdout
    cuda_mapping = cuda_bus_map() if cuda_mapping is None else cuda_mapping
    seen = set()
    gpus = []
    for record in csv.reader(io.StringIO(output), skipinitialspace=True):
        if not record or not any(x.strip() for x in record):
            continue
        if len(record) != 5:
            raise ValueError(f"Unexpected GPU inventory record: {record!r}")
        nvml_index, uuid, pci_bus_id, name, memory_mib = [x.strip() for x in record]
        if not uuid.startswith("GPU-") or uuid in seen:
            raise ValueError(f"Duplicate or invalid physical GPU UUID: {uuid!r}")
        seen.add(uuid)
        cuda_index = cuda_mapping.get(_bus_id(pci_bus_id))
        gpus.append({
            "uuid": uuid, "nvml_index": int(nvml_index),
            "cuda_index": cuda_index, "pci_bus_id": pci_bus_id,
            "model": name, "memory_mib": int(float(memory_mib)),
            "profiling_available": cuda_index is not None,
        })
    return sorted(gpus, key=lambda g: g["pci_bus_id"])


def discover_mig_mode():
    """Best-effort MIG status keyed by NVIDIA-SMI GPU index; unknown is explicit."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,mig.mode.current",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True)
        return {int(row[0].strip()): row[1].strip()
                for row in csv.reader(io.StringIO(result.stdout)) if len(row) == 2}
    except (OSError, subprocess.SubprocessError, ValueError):
        return {}


def read_gpu_health():
    """NVML-backed nvidia-smi GPU health snapshots, keyed by physical GPU UUID."""
    cmd = ["nvidia-smi", "--query-gpu=uuid,temperature.gpu,power.draw,power.limit,clocks.sm,clocks.mem,memory.used",
           "--format=csv,noheader,nounits"]
    try:
        output = subprocess.run(cmd, capture_output=True, text=True, timeout=10, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return {}

    def number(text):
        try:
            value = float(text.strip())
            import math
            return value if math.isfinite(value) else None
        except (ValueError, TypeError):
            return None

    found = {}
    for row in csv.reader(io.StringIO(output), skipinitialspace=True):
        if len(row) != 7:
            continue
        uid = row[0].strip()
        if uid.startswith("GPU-"):
            found[uid] = dict(zip(
                ("temperature_c", "power_w", "power_limit_w", "sm_clock_mhz",
                 "memory_clock_mhz", "memory_used_mib"),
                (number(x) for x in row[1:])))
    return found
