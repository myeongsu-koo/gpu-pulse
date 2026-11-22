import unittest

from precision_bench import MATRIX, VECTOR, NEEDS_BACKEND, status_row, supported_formats


class PrecisionBenchTests(unittest.TestCase):
    def test_catalog_has_all_requested_formats(self):
        catalog = supported_formats()
        self.assertEqual(len(catalog["Vector"]), 8)
        self.assertEqual(len(catalog["Matrix"]), 14)
        self.assertIn("NVFP4", MATRIX)
        self.assertIn("MXFP8", MATRIX)
        self.assertIn("INT4", VECTOR)

    def test_native_backend_requirements_are_explicit(self):
        for precision in ("INT1", "INT2", "INT4", "MXFP4",
                          "NVFP4", "MXFP6", "MXFP8"):
            self.assertIn(("Matrix", precision), NEEDS_BACKEND)

    def test_status_is_not_implicitly_measured(self):
        row = status_row("Matrix", "NVFP4", "BACKEND_REQUIRED", "missing backend")
        self.assertEqual(row["status"], "BACKEND_REQUIRED")
        self.assertNotIn("value", row)


if __name__ == "__main__":
    unittest.main()
