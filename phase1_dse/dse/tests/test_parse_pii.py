import tempfile
import textwrap
import unittest
from pathlib import Path

from dse.parse_pii import parse_pii, parse_pii_dir


SAMPLE_PII = textwrap.dedent(
    """
    t0: HBM[0] = bf16[64,64] Var['?arg0']()
    t1: HBM[8192] = bf16[64,64] Var['?arg1']()
    t2: D1[-1] = bf16[64,64] load_rm[rows='64'](t0)
    t3: D1[-1] = bf16[64,64] load_cm[rows='64'](t1)
    t4: D2[-1] = bf16[64,64] gemm(t2,t3)
    t5: D2[-1] = bf16[64,64] softmax[rows='64'](t4)
    t6: HBM[24576] = u8[8192] store_rm[rows='64'](t5)
    """
)


class ParsePiiTests(unittest.TestCase):
    def test_parse_single_pii(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "0.pii"
            p.write_text(SAMPLE_PII)
            prog = parse_pii(p)

            self.assertEqual(prog.kernel_name, "0")
            self.assertEqual(len(prog.instructions), 7)
            self.assertEqual(prog.instructions[2].name, "load_rm")
            self.assertEqual(prog.instructions[2].attrs["n"], 64)
            self.assertEqual(prog.instructions[4].name, "gemm")

    def test_parse_pii_dir_sort(self):
        with tempfile.TemporaryDirectory() as td:
            p1 = Path(td) / "1.pii"
            p0 = Path(td) / "0.pii"
            p1.write_text(SAMPLE_PII)
            p0.write_text(SAMPLE_PII)

            progs = parse_pii_dir(td)
            self.assertEqual([x.path.name for x in progs], ["0.pii", "1.pii"])


if __name__ == "__main__":
    unittest.main()
