"""Shared observer and supervisor defaults."""
DEFAULT_INTERVAL_MS = 100
PROFILER_BUSY_EXIT = 75


def profiler_busy(error):
    message = str(error).lower()
    return any(marker in message for marker in (
        "cupti_error_hardware_busy",
        "performance monitoring hardware is in use",
        "pm sampling is already enabled on the device",
        "pm sampling is already enabled for this device",
    ))
