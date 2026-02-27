import textwrap
import unittest

from dse.forward_bound import _load_yaml_with_fallback


class ForwardBoundConfigTests(unittest.TestCase):
    def test_yaml_subset_parse(self):
        txt = textwrap.dedent(
            """
            hardware:
              peak_compute_ops_per_s: 1.0e11
              bandwidth_bytes_per_s: 2.0e10
            sweep:
              bandwidth:
                start: 1.0e8
                stop: 1.0e11
                num: 25
                scale: log
            targets:
              latencies_s:
                - 1.0e-5
                - 5.0e-6
                - 2.0e-6
            """
        )
        cfg = _load_yaml_with_fallback(txt)
        self.assertEqual(cfg["sweep"]["bandwidth"]["scale"], "log")
        self.assertEqual(cfg["sweep"]["bandwidth"]["num"], 25)
        self.assertAlmostEqual(cfg["hardware"]["peak_compute_ops_per_s"], 1.0e11)
        self.assertEqual(len(cfg["targets"]["latencies_s"]), 3)
        self.assertAlmostEqual(cfg["targets"]["latencies_s"][1], 5.0e-6)


if __name__ == "__main__":
    unittest.main()
