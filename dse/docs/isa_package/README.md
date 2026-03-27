# ISA Package

This folder centralizes ISA/interface reference material used by the ACT hardware-mapping flow.

## Contents
- `taidl_primitive_reference.csv`: primitive-level TAIDL reference table.
- `taidl_xla_join_reference.csv`: join table between TAIDL-side and XLA-side views.
- `xla_hlo_operation_reference.csv`: HLO op reference used in the mapping story.
- `xla_taidl_hardware_class_view.csv`: HLO/TAIDL operations grouped by hardware abstraction class.
- `hardware_mapping_interface_spec.md`: end-to-end interface spec.
- `hardware_mapping_interface_example.json`: worked example of the mapping interface.

## Notes
- This folder is meant to be presentation-friendly: one place for ISA/interface references.
- Original source files may still exist in other repo locations.
