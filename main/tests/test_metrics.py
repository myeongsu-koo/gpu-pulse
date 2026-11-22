import unittest
from metrics import SPECS, choose_set, coverage, to_dashboard_values


class MetricTests(unittest.TestCase):
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

    def test_unavailable_remains_null(self):
        _, specs = choose_set({"one": ["dram__bytes_read"]})
        result = to_dashboard_values({specs[0].name: 10e9}, specs)
        self.assertEqual(result["dram_read_gbps"], 10.0)
        self.assertIsNone(result["dram_write_gbps"])
        self.assertIsNone(result["tensor_tflops"])
        self.assertFalse(coverage(specs)["dram_write"])


if __name__ == "__main__":
    unittest.main()
