"""Bounded live sample history and duration-weighted chart buckets."""
from collections import deque
from math import isfinite

HISTORY_SECONDS = 120
HISTORY_CAP = 16384
SECOND_NS = 1_000_000_000


class SampleHistory:
    def __init__(self, seconds=HISTORY_SECONDS, capacity=HISTORY_CAP):
        self.samples = deque(maxlen=capacity)
        self.age_ns = seconds * SECOND_NS

    def prune(self, now_ns):
        cutoff = now_ns - self.age_ns
        while self.samples and self.samples[0]["host_received_ns"] < cutoff:
            self.samples.popleft()


def average_buckets(samples, keys, width_ns=SECOND_NS, require_all=False):
    """Average rates within receipt-time buckets, weighted by sample duration."""
    buckets = {}
    for sample in samples:
        stamp = sample.get("host_received_ns")
        if not isinstance(stamp, int):
            continue
        bucket = buckets.setdefault(stamp // width_ns * width_ns, {})
        values = sample.get("measurements", {})
        if require_all and any(not valid_rate(values.get(key)) for key in keys):
            continue
        duration = sample.get("sample_duration_ms")
        weight = duration if valid_rate(duration) and duration > 0 else 1
        for key in keys:
            value = values.get(key)
            if valid_rate(value):
                total, elapsed = bucket.get(key, (0, 0))
                bucket[key] = total + value * weight, elapsed + weight
    return {stamp: {key: total / elapsed for key, (total, elapsed) in metrics.items()}
            for stamp, metrics in buckets.items()}


def valid_rate(value):
    return type(value) in (int, float) and isfinite(value) and value >= 0


def chart_samples(samples):
    from_keys = ("tensor_tflops", "cuda_fp32_tflops", "dram_read_gbps",
                 "dram_write_gbps", "l1_hit_pct", "l2_hit_pct")
    buckets = average_buckets(samples, from_keys, width_ns=SECOND_NS // 2)
    return [{"host_received_ns": stamp, "measurements": values}
            for stamp, values in sorted(buckets.items())]
