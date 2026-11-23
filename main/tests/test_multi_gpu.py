import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from gpu_discovery import _bus_id, discover_gpus
from node_aggregator import node_summary, compute_summary
from peak_registry import match_peak, attain, load_registry
from metrics import SPECS, precision_dashboard_values


def gpu(uid, model="NVIDIA L4", cuda=0, mib=24576):
    return {"uuid": uid, "model": model, "cuda_index": cuda,
            "pci_bus_id": f"00000000:0{cuda+1}:00.0", "memory_mib": mib}


class DiscoveryTests(unittest.TestCase):
    def test_pci_domain_normalization(self):
        self.assertEqual(_bus_id("00000000:65:00.0"), _bus_id("0000:65:00.0"))

    def test_cuda_ordinal_mapped_by_pci_not_nvml_index(self):
        sample = ("0, GPU-a, 00000000:65:00.0, NVIDIA L4, 23034\n"
                  "1, GPU-b, 00000000:41:00.0, NVIDIA L4, 23034\n")
        with patch("gpu_discovery.subprocess.run") as run:
            run.return_value.stdout = sample
            found = discover_gpus(cuda_mapping={"0000:65:00.0": 5, "0000:41:00.0": 4})
        self.assertEqual([g["cuda_index"] for g in found], [4, 5])
        self.assertEqual([g["uuid"] for g in found], ["GPU-b", "GPU-a"])

    def test_no_guess_if_cuda_bus_unmapped(self):
        with patch("gpu_discovery.subprocess.run") as run:
            run.return_value.stdout = "0, GPU-a, 00000000:65:00.0, NVIDIA L4, 23034\n"
            found = discover_gpus(cuda_mapping={})
        self.assertIsNone(found[0]["cuda_index"])
        self.assertFalse(found[0]["profiling_available"])

    def test_duplicate_gpu_uuid_rejected(self):
        with patch("gpu_discovery.subprocess.run") as run:
            run.return_value.stdout = ("0, GPU-a, 00000000:65:00.0, NVIDIA L4, 23034\n"
                                      "1, GPU-a, 00000000:66:00.0, NVIDIA L4, 23034\n")
            with self.assertRaises(ValueError):
                discover_gpus(cuda_mapping={})


class PeakTests(unittest.TestCase):
    def test_all_requested_families_in_registry(self):
        skus = load_registry()
        names = " ".join(" ".join(s["aliases"]) for s in skus)
        for family in ("T4", "L4", "L40S", "A100", "H100", "H200", "B200", "B300"):
            self.assertIn(family, names)

    def test_l4_does_not_match_l40s(self):
        p = match_peak(gpu("GPU-a", model="NVIDIA L40S", mib=49152))
        self.assertEqual(p["sku"], "l40s-48gb-pcie")

    def test_precise_l4_match(self):
        p = match_peak(gpu("GPU-a"))
        self.assertEqual(p["status"], "MATCHED")
        self.assertEqual(p["spec"]["compute_peaks"]["fp16.tensor.dense"]["value"], 121)

    def test_h100_without_form_factor_is_not_assumed_sxm(self):
        p = match_peak(gpu("GPU-a", model="NVIDIA H100 80GB HBM3", mib=81920))
        self.assertNotEqual(p["status"], "MATCHED")

    def test_peak_unknown_does_not_infer(self):
        self.assertEqual(match_peak(gpu("GPU-z", model="Unlisted NVIDIA GPU"))["status"], "SPEC_UNAVAILABLE")

    def test_attainment_validation(self):
        self.assertAlmostEqual(attain(48, 121), 39.6694214876)
        for value, peak in ((None, 100), (10, None), (10, 0), (-1, 10)):
            self.assertIsNone(attain(value, peak))


class AggregatorTests(unittest.TestCase):
    def setUp(self):
        self.devices = [gpu("GPU-a", cuda=0), gpu("GPU-b", cuda=1)]
        self.peak = {g["uuid"]: match_peak(g) for g in self.devices}
        self.now = 10_000_000_000
        self.latest = {
            "GPU-a": {"host_received_ns": self.now - 100_000_000,
                      "measurements": {"dram_read_gbps": 12.0, "dram_write_gbps": 3.0},
                      "precision_metrics": {"fp16.tensor.dense": {"value": 48.0, "unit": "TFLOP/s"}}},
            "GPU-b": {"host_received_ns": self.now - 200_000_000,
                      "measurements": {"dram_read_gbps": 10.0, "dram_write_gbps": 5.0},
                      "precision_metrics": {"fp16.tensor.dense": {"value": 60.0, "unit": "TFLOP/s"}}},
        }

    def test_complete_two_gpu_memory(self):
        result = node_summary(self.devices, self.latest, self.peak, now_ns=self.now)
        self.assertEqual(result["aggregation_status"], "COMPLETE")
        self.assertEqual(result["dram_read_gbps"], 22)
        self.assertEqual(result["dram_write_gbps"], 8)
        self.assertEqual(result["dram_attainment_pct"], 5.0)

    def test_partial_does_not_zero_fill_missing(self):
        result = node_summary(self.devices, {"GPU-a": self.latest["GPU-a"]},
                              self.peak, now_ns=self.now)
        self.assertEqual(result["aggregation_status"], "PARTIAL")
        self.assertEqual(result["sampled_memory_gpu_count"], 1)
        self.assertEqual(result["coverage_pct"], 50.0)

    def test_different_gpu_clock_domains_not_combined(self):
        far = dict(self.latest["GPU-b"], host_received_ns=self.now - 3_000_000_000)
        result = node_summary(self.devices, {"GPU-a": self.latest["GPU-a"], "GPU-b": far},
                              self.peak, now_ns=self.now)
        self.assertEqual(result["aggregation_status"], "PARTIAL")
        self.assertIsNone(result["dram_read_gbps"])

    def test_compute_sum_and_mixed_precision_isolation(self):
        result = compute_summary(self.devices, self.latest, self.peak,
                                 "fp16.tensor.dense", now_ns=self.now)
        self.assertEqual(result["status"], "COMPLETE")
        self.assertEqual(result["measured"], 108.0)
        self.assertEqual(result["peak"], 242.0)

    def test_cache_hit_not_averaged(self):
        for rec in self.latest.values():
            rec["measurements"]["l2_hit_pct"] = 80.0
        result = node_summary(self.devices, self.latest, self.peak, now_ns=self.now)
        self.assertNotIn("l2_hit_pct", result)


class PrecisionTests(unittest.TestCase):
    def test_precision_is_separate_from_aggregate_and_no_int8_guess(self):
        selected = [s for s in SPECS if s.key in ("tensor_fp16", "tensor_bf16")]
        vals = {s.name: (3e12 if s.key == "tensor_fp16" else 2e12) for s in selected}
        result = precision_dashboard_values(vals, selected)
        self.assertEqual(result["fp16.tensor.dense"]["value"], 3)
        self.assertEqual(result["bf16.tensor.dense"]["value"], 2)
        self.assertEqual(result["fp16.tensor.dense"]["comparison_status"], "DENSE_REFERENCE_UNVERIFIED")
        self.assertNotIn("int8.tensor.dense", result)


if __name__ == "__main__":
    unittest.main()
