import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from exporter_html import DATA_TAG, main, render_report


class ExporterTests(unittest.TestCase):
    def test_local_ndjson_exports_without_hardware_or_network(self):
        record = {"device": 0, "measurements": {"dram_read_gbps": 12},
                  "raw_metrics": {"payload": "</script><script>bad</script>"}}
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "capture.jsonl"
            report = Path(folder) / "report.html"
            source.write_text("\n".join(json.dumps(record) for _ in range(2)), encoding="utf-8")
            with (patch("socket.socket", side_effect=AssertionError("Network accessed")),
                  patch("ctypes.CDLL", side_effect=AssertionError("Driver loaded"))):
                main([str(source), "-o", str(report), "--model", "NVIDIA H200"])
            page = report.read_text(encoding="utf-8")

        payload = page.split(DATA_TAG, 1)[1].split("</script>", 1)[0]
        self.assertIn(r"\u003c/script>", payload)
        data = json.loads(payload)
        self.assertEqual(data["samples"], [record, record])
        self.assertEqual(data["sku"], "h200-141gb-sxm")
        self.assertEqual(data["collector"]["status"], "OFFLINE REPORT")

    def test_node_history_preserves_cuda_inventory(self):
        gpu = {"uuid": "GPU-a", "model": "NVIDIA L4", "memory_mib": 23034}
        page = render_report({"gpu": gpu, "samples": []})
        payload = page.split(DATA_TAG, 1)[1].split("</script>", 1)[0]
        self.assertEqual(json.loads(payload)["gpu"]["uuid"], "GPU-a")

    def test_mixed_cuda_devices_rejected(self):
        samples = [{"device": device, "measurements": {}} for device in (0, 1)]
        with self.assertRaisesRegex(ValueError, "one CUDA device"):
            render_report({"samples": samples})

    def test_embedded_inventory_and_compact_capture(self):
        metadata = {"uuid": "GPU-a", "model": "NVIDIA H200", "memory_mib": 143771}
        samples = [{"device": 0, "gpu": metadata, "host_received_ns": 1000 + index,
                    "measurements": {"dram_read_gbps": rate}}
                   for index, rate in enumerate((10, 20))]
        page = render_report({"samples": samples}, compact=True)
        payload = json.loads(page.split(DATA_TAG, 1)[1].split("</script>", 1)[0])
        self.assertEqual(payload["sku"], "h200-141gb-sxm")
        self.assertEqual(payload["gpu"], {**metadata, "cuda_index": 0})
        self.assertEqual(payload["samples"][0]["measurements"]["dram_read_gbps"], 15)
        self.assertEqual(len(payload["samples"]), 1)
