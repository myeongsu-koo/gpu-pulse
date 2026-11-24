"""Shared observer and supervisor defaults."""
DEFAULT_INTERVAL_MS = 100
PROFILER_BUSY_EXIT = 75


def profiler_busy(error):
    try:
        from cupti import cupti
    except (ImportError, OSError):
        cupti = None

    if cupti is not None:
        # The Python binding preserves CUptiResult in cuptiError.status.
        for cause in (error, error.__cause__):
            if isinstance(cause, cupti.cuptiError):
                return cause.status == cupti.Result.ERROR_HARDWARE_BUSY

    message = str(error).lower()
    return any(marker in message for marker in (
        "error_hardware_busy",
        "performance monitoring hardware is in use",
        "pm sampling is already enabled on the device",
        "pm sampling is already enabled for this device",
    ))
