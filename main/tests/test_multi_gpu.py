import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

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

    def test_driver_names_and_reported_memory_match(self):
        cases = [
            ("NVIDIA H100 80GB HBM3", 81559, "h100-80gb-sxm"),
            ("NVIDIA H200", 143771, "h200-141gb-sxm"),
            ("NVIDIA A100-SXM4-40GB", 40536, "a100-40gb-sxm"),
            ("NVIDIA-A100-SXM4-80GB", 81920, "a100-80gb-sxm"),
            ("NVIDIA A100-PCIE-40GB", 40536, "a100-40gb-pcie"),
            ("NVIDIA A100 80GB PCIe", 81920, "a100-80gb-pcie"),
            ("NVIDIA L40S", 46068, "l40s-48gb-pcie"),
            ("NVIDIA B300 SXM6 AC", 275040, "b300-hgx-sxm"),
            ("NVIDIA B300", 288 * 1024, "b300-hgx-sxm"),
            ("NVIDIA B200", 183359, "b200-hgx-sxm"),
            ("NVIDIA L4", 23034, "l4-24gb-pcie"),
            ("Tesla T4", 15360, "t4-16gb-pcie"),
        ]
        for model, mib, sku in cases:
            with self.subTest(model=model, mib=mib):
                self.assertEqual(match_peak(gpu("GPU-a", model=model, mib=mib))["sku"], sku)

    def test_variant_and_memory_checks_reject_incompatible_matches(self):
        cases = [
            ("NVIDIA H100", 81559),
            ("NVIDIA H100 PCIe", 81559),
            ("NVIDIA H100 NVL", 95830),
            ("NVIDIA H200 NVL", 143771),
            ("NVIDIA A100 80GB", 81920),
            ("NVIDIA A100-SXM4-40GB", 81920),
            ("NVIDIA L40S", 24000),
            ("NVIDIA B300", 183359),
        ]
        for model, mib in cases:
            with self.subTest(model=model, mib=mib):
                self.assertEqual(match_peak(gpu("GPU-a", model=model, mib=mib))["status"],
                                 "SPEC_UNAVAILABLE")

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


class NodeAgentTests(unittest.TestCase):
    def test_worker_launch_failure_is_isolated(self):
        from node_agent import NodeSupervisor
        inventory = [gpu("GPU-a"), gpu("GPU-b", cuda=1)]
        with tempfile.TemporaryDirectory() as folder:
            agent = NodeSupervisor(inventory, output_dir=folder)
            with patch("node_agent.subprocess.Popen", side_effect=OSError("permission denied")):
                agent.start()
            self.assertEqual(agent.worker_status()["GPU-a"]["status"], "FAILED")
            self.assertEqual(agent.worker_status()["GPU-b"]["status"], "FAILED")
            agent.shutdown()

    def test_summary_detail_and_registry_http(self):
        import threading
        import urllib.request
        from node_agent import NodeSupervisor, create_server
        from webui import PAGE
        with tempfile.TemporaryDirectory() as folder:
            agent = NodeSupervisor([gpu("GPU-a")], output_dir=folder)
            httpd = create_server(agent, host="127.0.0.1", port=0)
            thread = threading.Thread(target=httpd.serve_forever, daemon=True)
            thread.start()
            try:
                origin = f"http://127.0.0.1:{httpd.server_port}"
                def read(path):
                    with urllib.request.urlopen(origin + path, timeout=3) as response:
                        return json.load(response)
                self.assertEqual(read("/api/v1/node/summary")["gpu_count"], 1)
                self.assertEqual(read("/api/v1/gpus")["gpus"][0]["uuid"], "GPU-a")
                self.assertEqual(read("/api/v1/gpus/GPU-a/metrics")["sku"], "l4-24gb-pcie")
                with urllib.request.urlopen(origin + "/gpu/GPU-a", timeout=3) as response:
                    self.assertEqual(response.read(), PAGE.read_bytes())
                self.assertEqual(read("/api/v1/gpus/GPU-a/history")["samples"], [])
                self.assertEqual(read("/api/v1/peak-specs/b300-hgx-sxm")["sku"], "b300-hgx-sxm")
            finally:
                httpd.shutdown()
                httpd.server_close()
                thread.join(timeout=3)


class StandaloneDashboardTests(unittest.TestCase):
    def test_shared_page_and_legacy_state(self):
        import urllib.request
        from webui import PAGE, start_server

        sample = {"measurements": {"dram_read_gbps": 12},
                  "precision_metrics": {"fp16.tensor.dense": {"value": 60.5, "unit": "TFLOP/s"}}}
        snapshot = {"status": {"device": 1, "state": "running"}, "samples": [sample]}
        with (patch("webui.discover_gpus", return_value=[gpu("GPU-b", cuda=1)]) as inventory,
              patch("webui.read_gpu_health", return_value={"GPU-b": {"temperature_c": 29}})):
            state = Mock()
            state.snapshot.return_value = snapshot
            server = start_server(state, port=0)
            try:
                def read(path):
                    with urllib.request.urlopen(f"http://127.0.0.1:{server.server_port}{path}", timeout=3) as response:
                        return response.read()

                self.assertEqual(read("/"), PAGE.read_bytes())
                self.assertEqual(read("/dashboard.html"), PAGE.read_bytes())
                self.assertEqual(json.loads(read("/api/state")), snapshot)
                detail = json.loads(read("/api/detail"))
                self.assertEqual(detail["sku"], "l4-24gb-pcie")
                self.assertEqual(detail["compute"]["fp16.tensor.dense"]["attainment_pct"], 50)
                self.assertEqual(detail["health"]["temperature_c"], 29)
                self.assertEqual(detail["samples"], [sample])
                inventory.assert_called_once()
            finally:
                server.shutdown()
                server.server_close()


if __name__ == "__main__":
    unittest.main()
