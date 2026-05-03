# PrimeTime PX — small post-run sanity reports (text only).
# Run from the synopsys-pt-power step directory AFTER the step completes:
#   pt_shell -f /path/to/pt_post_sanity.tcl
# Or: bash run_pt_post_sanity.sh (wrapper sets cwd).

if {![file exists outputs/primetime.session]} {
  echo "ERROR: outputs/primetime.session not found. cd to the synopsys-pt-power step directory."
  exit 1
}

restore_session outputs/primetime.session

file mkdir reports/sanity

# Unannotated nets (should be near zero for good SAIF annotation)
redirect -file reports/sanity/switching_not_annotated.rpt {
  report_switching_activity -list_not_annotated
}

# Leakage vs dynamic split (compact)
redirect -file reports/sanity/leakage_only.rpt {
  report_power -leakage_only -nosplit
}

redirect -file reports/sanity/power_summary.rpt {
  report_power -nosplit
}

echo "Wrote reports/sanity/{switching_not_annotated,leakage_only,power_summary}.rpt"

exit
