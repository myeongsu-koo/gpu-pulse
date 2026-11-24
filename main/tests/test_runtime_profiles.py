import unittest
from unittest.mock import Mock

from metrics import KEY_TO_SPEC, direct_plans, precision_dashboard_values, to_dashboard_values
from node_aggregator import compute_summary
from observer import configure_plan
from webui import missing_measurements


class RuntimeProfiles(unittest.TestCase):
    def test_raw_sum_and_precision_table_agree(self):
        specs = next(direct_plans())
        values = {spec.name: 5e9 for spec in specs}
        dashboard = to_dashboard_values(values, specs, 0.01)
        table = precision_dashboard_values(values, specs, 0.01)
        self.assertEqual(dashboard["tensor_tflops"], 0.5)
        self.assertEqual(dashboard["cuda_fp32_tflops"], 2.0)
        self.assertEqual(table["fp16.tensor.dense"]["value"], 0.5)
        self.assertFalse(table["fp16.tensor.dense"]["counter_unit_verified"])
        self.assertIsNone(to_dashboard_values(values, specs, 0)["tensor_tflops"])
        self.assertEqual(dashboard["dram_read_gbps"], 5.0)

    def test_rejected_compute_probes_return_to_named_memory(self):
        collector = Mock()
        specs = [KEY_TO_SPEC["dram_read"]]

        def configure(**options):
            if options["single_pass_metric_set_name"] is None:
                raise ValueError("unsupported")

        collector.configure.side_effect = configure
        self.assertEqual(configure_plan(collector, ("named", specs), ["dram_read"], False, True),
                         ("named", specs))
        for call in collector.configure.call_args_list:
            self.assertIn(specs[0].name, call.kwargs["metrics"])

    def test_busy_is_not_retried(self):
        collector = Mock()
        collector.configure.side_effect = RuntimeError("ERROR_HARDWARE_BUSY")
        with self.assertRaisesRegex(RuntimeError, "HARDWARE_BUSY"):
            configure_plan(collector, None, [], False, True)
        collector.configure.assert_called_once()

    def test_missing_reasons_preserve_zero_and_old_samples(self):
        sample = {"selected_keys": ["dram_read"], "measurements": {"dram_read_gbps": 0}}
        reasons = missing_measurements(sample, {}, False)
        self.assertNotIn("dram_read_gbps", reasons)
        self.assertEqual(reasons["tensor_tflops"], "Outside selected counter set")
        self.assertEqual(missing_measurements(sample, {}, True)["dram_read_gbps"], "Stale sample")
        self.assertEqual(missing_measurements({"measurements": {}}, {}, False)["tensor_tflops"],
                         "No valid reading")

    def test_unknown_peak_does_not_hide_verified_node_rate(self):
        key = "fp32.cuda.dense"
        sample = {"host_received_ns": 10, "precision_metrics": {key: {"unit": "TFLOP/s", "value": 3}}}
        result = compute_summary([{"uuid": "a"}], {"a": sample}, {}, key, now_ns=10)
        self.assertEqual(result["measured"], 3)
        self.assertIsNone(result["attainment_pct"])
