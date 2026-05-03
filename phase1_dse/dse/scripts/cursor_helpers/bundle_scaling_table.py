#!/usr/bin/env python3
"""Summarize multiple pt_comparison_bundle.json files for scaling / trend review.

Scans .cursor/docs/artifacts/runs/**/pt_comparison_bundle.json (excluding _template),
writes a small CSV: workload_label, bundle path, pct_error_vs_pt, window energy, etc.

No large binary output.
"""

import csv
import json
import sys
from pathlib import Path


def main():
    submodule = Path(__file__).resolve().parents[2]
    root = submodule / ".cursor" / "docs" / "artifacts" / "runs"
    rows = []
    for p in sorted(root.glob("**/pt_comparison_bundle.json")):
        if "_template" in p.parts:
            continue
        try:
            data = json.loads(p.read_text())
        except Exception as e:
            sys.stderr.write("skip {}: {}\n".format(p, e))
            continue
        act = data.get("act") or {}
        win = data.get("window") or {}
        rows.append(
            {
                "workload_label": data.get("workload_label", ""),
                "dut_scope": data.get("dut_scope", ""),
                "bundle_relpath": str(p.relative_to(submodule)),
                "pt_window_energy_uj": win.get("energy_uj", ""),
                "act_total_energy_uj": act.get("total_energy_uj", ""),
                "pct_error_vs_pt": act.get("pct_error_vs_pt", ""),
                "capture_ns": (win.get("capture_ns", "")),
            }
        )

    out = root / "scaling_summary.csv"
    if not rows:
        sys.stderr.write("No bundles found under {}\n".format(root))
        sys.exit(1)

    with out.open("w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("Wrote {} ({} rows)\n".format(out, len(rows)))
    for r in rows:
        print("{workload_label:30} pct_err={pct_error_vs_pt}".format(**r))


if __name__ == "__main__":
    main()
