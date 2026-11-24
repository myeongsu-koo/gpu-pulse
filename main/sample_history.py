"""Bounded live sample history and duration-weighted chart buckets."""
from collections import deque
if __package__:
    from .sample_schema import SECOND_NS, nonnegative_number
else:
    from sample_schema import SECOND_NS, nonnegative_number

HISTORY_SECONDS = 120
HISTORY_CAP = 16384


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
        if type(stamp) is not int:
            continue
        bucket = buckets.setdefault(stamp // width_ns * width_ns, {})
        values = sample.get("measurements", {})
        if require_all and any(nonnegative_number(values.get(key)) is None for key in keys):
            continue
        weight = nonnegative_number(sample.get("sample_duration_ms")) or 1
        for key in keys:
            value = nonnegative_number(values.get(key))
            if value is not None:
                total, elapsed = bucket.get(key, (0, 0))
                bucket[key] = total + value * weight, elapsed + weight
    result = {}
    for stamp, metrics in buckets.items():
        averages = {
            key: value for key, (total, elapsed) in metrics.items()
            if (value := nonnegative_number(total / elapsed)) is not None
        }
        result[stamp] = averages if not require_all or len(averages) == len(keys) else {}
    return result


def chart_samples(samples):
    from_keys = ("tensor_tflops", "cuda_fp32_tflops", "dram_read_gbps",
                 "dram_write_gbps", "l1_hit_pct", "l2_hit_pct")
    buckets = average_buckets(samples, from_keys, width_ns=SECOND_NS // 2)
    return [{"host_received_ns": stamp, "measurements": values}
            for stamp, values in sorted(buckets.items())]
