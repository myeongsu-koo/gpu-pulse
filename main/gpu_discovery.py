"""Inventory CUDA-visible devices through libcuda, using device ordinals directly."""
from __future__ import annotations

import ctypes
import re
from uuid import UUID


def discover_gpus(driver=None):
    """Return CUDA-visible devices without creating a context or loading NVML."""
    if driver is None:
        driver = ctypes.CDLL("libcuda.so.1")

    def call(symbol, argtypes, *args):
        fn = getattr(driver, symbol)
        fn.argtypes = argtypes
        fn.restype = ctypes.c_int
        result = fn(*args)
        if result:
            raise RuntimeError(f"CUDA Driver {symbol} returned {result}")

    integer = ctypes.c_int
    integer_ptr = ctypes.POINTER(integer)
    string_args = [ctypes.c_char_p, integer, integer]
    memory_type = ctypes.c_size_t
    uuid_type = ctypes.c_ubyte * 16
    count = integer()
    # CUDA_ERROR_NO_DEVICE is 100. A host without devices can still serve an empty node.
    driver.cuInit.argtypes = [ctypes.c_uint]
    driver.cuInit.restype = integer
    init = driver.cuInit(0)
    if init == 100:
        return []
    if init:
        raise RuntimeError(f"CUDA Driver cuInit returned {init}")
    call("cuDeviceGetCount", [integer_ptr], ctypes.byref(count))
    gpus = []
    for ordinal in range(count.value):
        device = integer()
        name = ctypes.create_string_buffer(256)
        bus = ctypes.create_string_buffer(32)
        memory = memory_type()
        device_uuid = uuid_type()
        call("cuDeviceGet", [integer_ptr, integer], ctypes.byref(device), ordinal)
        call("cuDeviceGetName", string_args, name, len(name), device)
        call("cuDeviceGetPCIBusId", string_args, bus, len(bus), device)
        call("cuDeviceTotalMem_v2", [ctypes.POINTER(memory_type), integer], ctypes.byref(memory), device)
        call("cuDeviceGetUuid_v2", [ctypes.POINTER(uuid_type), integer], ctypes.byref(device_uuid), device)
        model = name.value.decode("ascii", errors="replace")
        mig = bool(re.search(r"\bMIG\b", model, re.IGNORECASE))
        gpus.append({
            "uuid": ("MIG-" if mig else "GPU-") + str(UUID(bytes=bytes(device_uuid))),
            "cuda_index": ordinal, "pci_bus_id": bus.value.decode("ascii"),
            "model": model, "memory_mib": memory.value // (1024 * 1024),
            "profiling_available": not mig,
        })
    return gpus
