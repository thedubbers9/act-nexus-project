import json
import tempfile
import textwrap
import unittest
from pathlib import Path

from dse.forward_bound import run_forward_bound


SAMPLE_PII = textwrap.dedent(
    """
    t0: HBM[0] = bf16[64,64] Var['?arg0']()
    t1: HBM[8192] = bf16[64,64] Var['?arg1']()
    t2: HBM[16384] = bf16[64,64] Var['?arg2']()
    t3: D1[-1] = bf16[64,64] load_rm[rows='64'](t0)
    t4: D1[-1] = bf16[64,64] load_cm[rows='64'](t1)
    t5: D2[-1] = bf16[64,64] gemm(t3,t4)
    t6: D2[-1] = bf16[64,64] softmax[rows='64'](t5)
    t7: D1[-1] = bf16[64,64] mov[rows='64'](t6)
    t8: D1[-1] = bf16[64,64] load_rm[rows='64'](t2)
    t9: D2[-1] = bf16[64,64] gemm(t7,t8)
    t10: HBM[24576] = u8[8192] store_rm[rows='64'](t9)
    """
)


class ForwardBoundPiiIntegrationTests(unittest.TestCase):
    def test_end_to_end_pii(self):
        repo_root = Path(__file__).resolve().parents[2]
        config_file = repo_root / "dse" / "config.yaml"

        with tempfile.TemporaryDirectory() as td:
            inp = Path(td) / "0.pii"
            inp.write_text(SAMPLE_PII)
            out = Path(td) / "out"

            summary = run_forward_bound(
                input_path=str(inp),
                hw_config_path=str(config_file),
                targets_config_path=None,
                out_dir=str(out),
                with_plot=False,
                input_mode="pii",
            )

            self.assertEqual(summary["status"], "ok")
            self.assertEqual(summary["input_mode_used"], "pii")
            self.assertTrue((out / "forward_bounds.csv").exists())
            self.assertTrue((out / "frontier.csv").exists())
            self.assertTrue((out / "summary.json").exists())

            saved = json.loads((out / "summary.json").read_text())
            self.assertEqual(saved["input_mode_used"], "pii")
            self.assertGreater(saved["num_eval_rows"], 0)


if __name__ == "__main__":
    unittest.main()
