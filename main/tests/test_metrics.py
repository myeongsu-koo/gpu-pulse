import unittest
from unittest.mock import patch
from metrics import SPECS, choose_set, coverage, to_dashboard_values
from observer import capability_report


class MetricTests(unittest.TestCase):
    def test_l4_qualified_cupti_metric_sets(self):
        metrics = {
            "FBSP": ("dramc__sectors_op_read", "dramc__sectors_op_write"),
            "SM_B": ("l1tex__t_sector_hit_rate",),
            "LTS": ("lts__average_t_sector_hit_rate_realtime",),
        }
        sets = {
            name: [f"{domain}.{name}.{metric}"
                   for domain, bases in metrics.items() for metric in bases]
            for name in ("TriageAC", "TriageSCG")
        }
        expected = [
            "dramc__sectors_op_read.sum",
            "dramc__sectors_op_write.sum",
            "l1tex__t_sector_hit_rate.pct",
            "lts__average_t_sector_hit_rate_realtime.pct",
        ]
        name, specs = choose_set(sets, required=["dram_read_sectors"])
        self.assertEqual(name, "TriageAC")
        self.assertEqual([spec.name for spec in specs], expected)

        with patch("observer.discover_sets", return_value=sets):
            report = capability_report("AD104")
        self.assertEqual(report["preferred_plan"]["metric_names"], expected)
        self.assertEqual(report["candidate_availability"]["dram_read_sectors"]["sets_with_base"],
                         ["TriageAC", "TriageSCG"])
        self.assertEqual(report["candidate_availability"]["tensor_fp16"]["sets_with_base"], [])

    def test_no_fp32_guess_from_partial(self):
        _, specs = choose_set({"a": ["smsp__sass_thread_inst_executed_op_ffma_pred_on"]})
        metrics = {specs[0].name: 1e12}
        result = to_dashboard_values(metrics, specs)
        self.assertIsNone(result["cuda_fp32_tflops"])

    def test_throughput_units(self):
        names = [s.base for s in SPECS]
        _, specs = choose_set({"all": names})
        vals = {s.name: 0.0 for s in specs}
        vals[next(s.name for s in specs if s.key == "tensor_bf16")] = 9e12
        vals[next(s.name for s in specs if s.key == "fadd")] = 1e12
        vals[next(s.name for s in specs if s.key == "fmul")] = 2e12
        vals[next(s.name for s in specs if s.key == "ffma")] = 3e12
        vals[next(s.name for s in specs if s.key == "dram_read")] = 800e9
        vals[next(s.name for s in specs if s.key == "l2_hit")] = 89.3
        result = to_dashboard_values(vals, specs)
        self.assertEqual(result["tensor_tflops"], 9.0)
        self.assertEqual(result["cuda_fp32_tflops"], 9.0)
        self.assertEqual(result["dram_read_gbps"], 800.0)
        self.assertEqual(result["l2_hit_pct"], 89.3)

    def test_l4_cache_and_sector_fallbacks(self):
        _, specs = choose_set({"l4": [
            "dramc__sectors_op_read", "dramc__sectors_op_write",
            "l1tex__t_sector_hit_rate",
            "lts__average_t_sector_hit_rate_realtime",
        ]})
        vals = {s.name: 0.0 for s in specs}
        vals[next(s.name for s in specs if s.key == "dram_read_sectors")] = 2_000_000
        vals[next(s.name for s in specs if s.key == "dram_write_sectors")] = 500_000
        vals[next(s.name for s in specs if s.key == "l1_hit")] = 17.5
        vals[next(s.name for s in specs if s.key == "l2_hit_realtime")] = 94.0
        result = to_dashboard_values(vals, specs, sample_seconds=0.1)
        self.assertAlmostEqual(result["dram_read_gbps"], 0.64)
        self.assertAlmostEqual(result["dram_write_gbps"], 0.16)
        self.assertEqual(result["l1_hit_pct"], 17.5)
        self.assertEqual(result["l2_hit_pct"], 94.0)
        self.assertTrue(coverage(specs)["l2_hit"])

    def test_sector_rates_need_sample_duration(self):
        _, specs = choose_set({"l4": ["dramc__sectors_op_read"]})
        result = to_dashboard_values({specs[0].name: 100}, specs)
        self.assertIsNone(result["dram_read_gbps"])

    def test_sector_duration_edge_cases(self):
        _, specs = choose_set({"l4": [
            "dramc__sectors_op_read", "dramc__sectors_op_write"
        ]})
        values = {
            next(s.name for s in specs if s.key == "dram_read_sectors"): 100,
            next(s.name for s in specs if s.key == "dram_write_sectors"): 0,
        }
        for invalid in (None, 0, -1, float("nan"), float("inf"), -float("inf")):
            with self.subTest(duration=invalid):
                result = to_dashboard_values(values, specs, sample_seconds=invalid)
                self.assertIsNone(result["dram_read_gbps"])
                self.assertIsNone(result["dram_write_gbps"])

        actual = to_dashboard_values(values, specs, sample_seconds=0.5)
        self.assertAlmostEqual(actual["dram_read_gbps"], 100 * 32 / 0.5 / 1e9)
        self.assertEqual(actual["dram_write_gbps"], 0.0)

    def test_invalid_sector_values_remain_null(self):
        _, specs = choose_set({"l4": ["dramc__sectors_op_read"]})
        for invalid in (None, -1, float("nan"), float("inf"), "unavailable"):
            with self.subTest(value=invalid):
                result = to_dashboard_values({specs[0].name: invalid}, specs, sample_seconds=0.1)
                self.assertIsNone(result["dram_read_gbps"])

    def test_primary_dram_rates_take_precedence_over_sectors(self):
        _, specs = choose_set({"both": [
            "dram__bytes_read", "dram__bytes_write",
            "dramc__sectors_op_read", "dramc__sectors_op_write"
        ]})
        values = {s.name: 0.0 for s in specs}
        values[next(s.name for s in specs if s.key == "dram_read")] = 30e9
        values[next(s.name for s in specs if s.key == "dram_write")] = 15e9
        values[next(s.name for s in specs if s.key == "dram_read_sectors")] = 1e9
        values[next(s.name for s in specs if s.key == "dram_write_sectors")] = 1e9
        result = to_dashboard_values(values, specs, sample_seconds=0.1)
        self.assertEqual(result["dram_read_gbps"], 30.0)
        self.assertEqual(result["dram_write_gbps"], 15.0)

    def test_invalid_primary_falls_back_to_sector_rate(self):
        _, specs = choose_set({"both": [
            "dram__bytes_read", "dramc__sectors_op_read"
        ]})
        values = {s.name: 100 for s in specs}
        values[next(s.name for s in specs if s.key == "dram_read")] = float("nan")
        result = to_dashboard_values(values, specs, sample_seconds=0.2)
        self.assertAlmostEqual(result["dram_read_gbps"], 100 * 32 / 0.2 / 1e9)

    def test_primary_l2_hit_takes_precedence_and_invalid_falls_back(self):
        _, specs = choose_set({"both": [
            "lts__t_sector_hit_rate", "lts__average_t_sector_hit_rate_realtime"
        ]})
        values = {s.name: 0.0 for s in specs}
        primary = next(s.name for s in specs if s.key == "l2_hit")
        realtime = next(s.name for s in specs if s.key == "l2_hit_realtime")
        values[primary], values[realtime] = 65.0, 97.0
        self.assertEqual(to_dashboard_values(values, specs)["l2_hit_pct"], 65.0)
        values[primary] = None
        self.assertEqual(to_dashboard_values(values, specs)["l2_hit_pct"], 97.0)

    def test_coverage_marks_fallbacks_available(self):
        _, specs = choose_set({"l4": [
            "dramc__sectors_op_read", "dramc__sectors_op_write",
            "lts__average_t_sector_hit_rate_realtime",
        ]})
        available = coverage(specs)
        self.assertTrue(available["dram_read"])
        self.assertTrue(available["dram_write"])
        self.assertTrue(available["l2_hit"])
        self.assertFalse(available["l1_hit"])
        # Coverage describes selected capabilities, not a successfully normalized value.
        values = {s.name: 100 for s in specs}
        self.assertIsNone(to_dashboard_values(values, specs)["dram_read_gbps"])

    def test_unavailable_remains_null(self):
        _, specs = choose_set({"one": ["dram__bytes_read"]})
        result = to_dashboard_values({specs[0].name: 10e9}, specs)
        self.assertEqual(result["dram_read_gbps"], 10.0)
        self.assertIsNone(result["dram_write_gbps"])
        self.assertIsNone(result["tensor_tflops"])
        self.assertFalse(coverage(specs)["dram_write"])


if __name__ == "__main__":
    unittest.main()
