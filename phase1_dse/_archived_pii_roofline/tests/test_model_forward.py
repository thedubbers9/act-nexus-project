import unittest

from dse.features import FeatureRow
from dse.model_forward import evaluate_row, inverse_bandwidth_bounds


class ModelTests(unittest.TestCase):
    def test_evaluate_row_memory_bound(self):
        feat = FeatureRow(
            candidate="0.py",
            kernel_name="k",
            instruction_count=1,
            hbm_read_bytes=1000,
            hbm_write_bytes=0,
            hbm_bytes=1000,
            local_bytes=0,
            compute_ops=100,
            intensity_ops_per_hbm_byte=0.1,
            unknown_instruction_count=0,
            unknown_instructions="",
        )
        row = evaluate_row(feat, bandwidth_bytes_per_s=500, peak_compute_ops_per_s=100)
        self.assertAlmostEqual(row.t_mem_lb_s, 2.0)
        self.assertAlmostEqual(row.t_cmp_lb_s, 1.0)
        self.assertEqual(row.bottleneck, "memory")

    def test_inverse_bandwidth(self):
        feat = FeatureRow(
            candidate="0.py",
            kernel_name="k",
            instruction_count=1,
            hbm_read_bytes=1000,
            hbm_write_bytes=0,
            hbm_bytes=1000,
            local_bytes=0,
            compute_ops=100,
            intensity_ops_per_hbm_byte=0.1,
            unknown_instruction_count=0,
            unknown_instructions="",
        )
        rows = inverse_bandwidth_bounds([feat], [2.0], peak_compute_ops_per_s=100)
        self.assertEqual(rows[0]["status"], "feasible")
        self.assertAlmostEqual(rows[0]["min_required_bandwidth_bytes_per_s"], 500.0)


if __name__ == "__main__":
    unittest.main()
