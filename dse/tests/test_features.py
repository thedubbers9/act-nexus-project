import tempfile
import textwrap
import unittest
from pathlib import Path

from dse.features import extract_features
from dse.parse_pii import parse_pii


SAMPLE = textwrap.dedent(
    """
    t0: HBM[0] = bf16[64,64] Var['?arg0']()
    t1: HBM[8192] = bf16[64,64] Var['?arg1']()
    t2: D1[-1] = bf16[64,64] load_rm[rows='64'](t0)
    t3: D1[-1] = bf16[64,64] load_cm[rows='64'](t1)
    t4: D2[-1] = bf16[64,64] gemm(t2,t3)
    t5: HBM[24576] = u8[8192] store_rm[rows='64'](t4)
    """
)


class FeatureTests(unittest.TestCase):
    def test_feature_values(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "0.pii"
            p.write_text(SAMPLE)
            prog = parse_pii(p)
            feat = extract_features(prog)

            # 2 loads + 1 store, each 64*64*2 = 8192 bytes
            self.assertEqual(feat.hbm_read_bytes, 16384)
            self.assertEqual(feat.hbm_write_bytes, 8192)
            self.assertEqual(feat.hbm_bytes, 24576)
            self.assertGreater(feat.compute_ops, 0)
            self.assertEqual(feat.unknown_instruction_count, 0)


if __name__ == "__main__":
    unittest.main()
