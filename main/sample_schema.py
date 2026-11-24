"""Validation shared by the collector, worker stream and offline reports."""
import json
import math
from numbers import Real

SECOND_NS = 1_000_000_000
MAX_SAMPLE_AGE_S = 5.0


def nonnegative_number(value):
    """Return a finite nonnegative float, or None for a missing/invalid value."""
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    try:
        number = float(value)
    except OverflowError:
        return None
    return number if math.isfinite(number) and number >= 0 else None


def sample_age_seconds(sample, now_ns):
    stamp = sample.get("host_received_ns")
    if type(stamp) is not int or stamp < 0 or stamp > now_ns:
        return None
    return (now_ns - stamp) / SECOND_NS


def decode_json(source):
    def finite_float(token):
        value = float(token)
        if not math.isfinite(value):
            raise ValueError(f"Non-finite JSON number: {token}")
        return value

    return json.loads(source, parse_float=finite_float, parse_constant=finite_float)


def validate_sample(sample):
    """Check the sample envelope while allowing older captures and extra metrics."""
    if not isinstance(sample, dict) or not isinstance(sample.get("measurements"), dict):
        raise ValueError("Expected observer sample with a measurements object")
    for key in ("precision_metrics", "raw_metrics", "gpu"):
        if key in sample and not isinstance(sample[key], dict):
            raise ValueError(f"Expected sample {key} to be an object")
    for key in ("device", "host_received_ns", "gpu_start_ns", "gpu_end_ns"):
        value = sample.get(key)
        if value is not None and (type(value) is not int or value < 0):
            raise ValueError(f"Expected sample {key} to be a nonnegative integer")
