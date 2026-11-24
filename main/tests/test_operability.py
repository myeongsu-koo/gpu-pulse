import argparse
import contextlib
import io
import json
import sys
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from node_agent import NodeSupervisor, create_server
from observer import State, run
from peak_registry import match_peak
from sample_history import SECOND_NS
from webui import detail_payload, start_server


def gpu(uid, index=0, available=True):
    return dict(uuid=uid, model="Tesla T4", cuda_index=index, memory_mib=15360,
                profiling_available=available)


class HistoryTests(unittest.TestCase):
    def test_observer_expires_old_samples_and_keeps_count_cap(self):
        state = State(seconds=3, capacity=2)
        now = 10 * SECOND_NS
        for age in (5, 2, 1, 0):
            state.push({"host_received_ns": now - age * SECOND_NS})
        with patch("observer.time.time_ns", return_value=now):
            self.assertEqual(len(state.snapshot()["samples"]), 2)
        with patch("observer.time.time_ns", return_value=now + 4 * SECOND_NS):
            self.assertEqual(state.snapshot()["samples"], [])

    def test_node_averages_every_sample_and_preserves_partial_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            agent = NodeSupervisor([gpu("a"), gpu("b", 1, False)], output_dir=directory, history_seconds=4)
            second = time.time_ns() // SECOND_NS * SECOND_NS - SECOND_NS
            history = agent.history["a"].samples
            for stamp, duration, read in ((second - 5 * SECOND_NS, 1, 100), (second, 1, 10), (second + 1, 3, 30)):
                history.append({"host_received_ns": stamp, "sample_duration_ms": duration,
                                "measurements": {"dram_read_gbps": read, "dram_write_gbps": 5}})
            points = agent.node_history()["points"]
            self.assertEqual(len(points), 1)
            self.assertEqual(points[0]["dram_read_gbps"], 25)
            self.assertEqual(points[0]["status"], "PARTIAL")
            self.assertEqual(points[0]["sampled_gpus"], 1)


class CollectorTests(unittest.TestCase):
    def test_final_drain_decodes_more_than_one_batch(self):
        collector = Mock()
        batch = type("Batch", (list,), {"metrics": []})
        collector.decode.side_effect = [
            batch([SimpleNamespace(start_timestamp=i, end_timestamp=i + 1, metric_values=[])])
            for i in (1, 2)
        ] + [batch()]
        module = SimpleNamespace(Collector=lambda **_: collector,
                                 TriggerMode=SimpleNamespace(GPU_TIME_INTERVAL=SimpleNamespace(name="TIME")),
                                 HardwareBufferAppendMode=SimpleNamespace(KEEP_LATEST=0))
        args = argparse.Namespace(device=0, history_seconds=120, probe=False, require="", metric_set=None,
                                  interval_ms=10, hardware_buffer_mb=1, web=False, output="-",
                                  poll_ms=1, duration=1, max_samples=1)
        stop = Mock()
        stop.wait.return_value = True
        out = io.StringIO()
        with (patch.dict(sys.modules, {"cupti": SimpleNamespace(pm_sampling=module)}),
              patch("observer.threading.Event", return_value=stop), patch("observer.signal.signal"),
              patch("observer.CudaPrimaryContext", return_value=contextlib.nullcontext()),
              patch("observer.discover_gpus", return_value=[gpu("GPU-test")]),
              patch("observer.get_chip_name", return_value="AD104"),
              patch("observer.select_plan", return_value=("test", [])),
              contextlib.redirect_stdout(out)):
            self.assertEqual(run(args), 0)
        self.assertEqual(collector.decode.call_count, 3)
        rows = [json.loads(line) for line in out.getvalue().splitlines()]
        self.assertEqual([row["gpu_end_ns"] for row in rows], [2, 3])
        self.assertEqual(rows[-1]["gpu"]["uuid"], "GPU-test")

    def test_restarts_failures_after_backoff_only(self):
        with tempfile.TemporaryDirectory() as directory:
            agent = NodeSupervisor([gpu("failed"), gpu("clean", 1)], output_dir=directory)
            agent.workers = {
                uid: {"process": Mock(poll=Mock(return_value=code)),
                      "thread": Mock(is_alive=Mock(return_value=False)), "delay": 2,
                      "started_at": 100, "status": "STARTING"}
                for uid, code in (("failed", 1), ("clean", 0))
            }
            with patch.object(agent, "_launch") as launch:
                with patch("node_agent.time.monotonic", return_value=101):
                    agent.restart_failed()
                    launch.assert_not_called()
                with patch("node_agent.time.monotonic", return_value=103):
                    agent.restart_failed()
                    launch.assert_called_once_with(agent.by_uuid["failed"], delay=4)

    def test_dashboard_rejects_external_binds(self):
        with self.assertRaisesRegex(ValueError, "127.0.0.1"):
            start_server(Mock(), host="0.0.0.0")
        with self.assertRaisesRegex(ValueError, "127.0.0.1"):
            create_server(Mock(), host="0.0.0.0")

    def test_t4_lower_bound_is_not_a_peak(self):
        card = gpu("t4")
        info = detail_payload(card, match_peak(card), {}, None, 0)
        self.assertIsNone(info["memory_bandwidth_peak_gbps"])

    def test_profiler_conflict_has_distinct_status_and_slow_retry(self):
        from collector_config import PROFILER_BUSY_EXIT
        from observer import main

        with patch("observer.run", side_effect=RuntimeError("CUPTI_ERROR_HARDWARE_BUSY")):
            self.assertEqual(main([]), PROFILER_BUSY_EXIT)
        with tempfile.TemporaryDirectory() as directory:
            agent = NodeSupervisor([gpu("a")], output_dir=directory)
            agent.workers["a"] = {
                "status": "STARTING", "process": Mock(poll=Mock(return_value=PROFILER_BUSY_EXIT)),
                "thread": Mock(is_alive=Mock(return_value=False)), "delay": 2, "started_at": 100,
            }
            with patch("node_agent.time.monotonic", return_value=102):
                agent.restart_failed()
                status = agent.worker_status()["a"]
            self.assertEqual(status["status"], "PROFILER_BUSY")
            self.assertEqual(status["retry_in_s"], 30)

    def test_shutdown_cleans_managed_log_directory(self):
        agent = NodeSupervisor([])
        self.assertTrue(agent.output_dir.exists())
        agent.shutdown()
        self.assertFalse(agent.output_dir.exists())

    def test_retry_preserves_previous_stderr(self):
        with tempfile.TemporaryDirectory() as directory:
            agent = NodeSupervisor([gpu("a")], output_dir=directory)
            log = agent.output_dir / "a.stderr.log"
            log.write_text("first failure", encoding="utf-8")
            with patch("node_agent.subprocess.Popen", side_effect=OSError("launch failed")):
                agent.start()
            self.assertEqual(log.with_suffix(".log.1").read_text(encoding="utf-8"), "first failure")
            agent.shutdown()


    def test_structured_cupti_result_overrides_exception_text(self):
        from collector_config import profiler_busy

        class CuptiError(Exception):
            def __init__(self, status, message):
                super().__init__(message)
                self.status = status

        binding = SimpleNamespace(cuptiError=CuptiError, Result=SimpleNamespace(ERROR_HARDWARE_BUSY=26))
        with patch.dict(sys.modules, {"cupti": SimpleNamespace(cupti=binding)}):
            self.assertTrue(profiler_busy(CuptiError(26, "opaque failure")))
            self.assertFalse(profiler_busy(CuptiError(1, "ERROR_HARDWARE_BUSY")))
            outer = RuntimeError("wrapper")
            outer.__cause__ = CuptiError(26, "opaque failure")
            self.assertTrue(profiler_busy(outer))
        self.assertTrue(profiler_busy(RuntimeError("ERROR_HARDWARE_BUSY (26)")))


    def test_invalid_cupti_numbers_are_json_null(self):
        from observer import emit_sample
        from metrics import KEY_TO_SPEC

        spec = KEY_TO_SPEC["dram_read"]
        sample = SimpleNamespace(start_timestamp=1, end_timestamp=2, metric_values=[float("nan")])
        record = emit_sample(sample, [spec.name], [spec], "AD104", 0)
        self.assertIsNone(record["raw_metrics"][spec.name])
        self.assertIsNone(record["measurements"]["dram_read_gbps"])
        json.dumps(record, allow_nan=False)

    def test_stale_live_detail_retains_reference_and_omits_old_reading(self):
        from webui import detail_payload

        card = gpu("a")
        record = {"host_received_ns": SECOND_NS, "measurements": {"dram_read_gbps": 10}}
        match = match_peak(card)
        live = detail_payload(card, match, record, None, 1, now_ns=7 * SECOND_NS)
        offline = detail_payload(card, match, record, None, 1)
        self.assertTrue(live["sample_stale"])
        self.assertEqual(live["measurements"], {})
        self.assertEqual(live["sku"], offline["sku"])
        self.assertEqual(offline["measurements"]["dram_read_gbps"], 10)

    def test_worker_rejects_malformed_samples_without_losing_stream(self):
        with tempfile.TemporaryDirectory() as directory:
            agent = NodeSupervisor([gpu("a")], output_dir=directory)
            valid = '{"device":0,"measurements":{"dram_read_gbps":7}}'
            pipe = io.StringIO('{"device":0,"measurements":[]}\n'
                               '{"device":0,"measurements":{"dram_read_gbps":1e999}}\n'
                               + valid + '\n')
            with self.assertLogs("gpu-pulse-node", level="WARNING") as captured:
                agent._reader("a", pipe)
            self.assertEqual(len(captured.records), 2)
            self.assertEqual(agent.latest["a"]["measurements"]["dram_read_gbps"], 7)
