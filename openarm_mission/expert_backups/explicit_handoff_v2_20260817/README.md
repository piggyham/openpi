# Explicit handoff v2 expert backup

This directory preserves the expert implementation used to collect
`p10_explicit_handoff_full` and train
`p10_explicit_handoff_v2_lora_bs1_scalar_20260817`.

The snapshots were taken before restoring the previous natural-hang expert on
2026-08-17. They contain the explicit sequence:

1. Open the right gripper.
2. Retreat vertically.
3. Hold the high completion pose for 1.2 seconds.
4. Stow the right arm before the left arm approaches.

Files:

- `friction_expert.py.snapshot`: complete expert source.
- `config.py.snapshot`: complete expert and mission configuration.
- `dataset.py.snapshot`: dataset schema and explicit-handoff version marker.
- `test_friction.py.snapshot`: matching expert test.

To restore this version, copy each snapshot over the corresponding file after
first reviewing any newer changes. The archived dataset version is
`openarm-p10-explicit-handoff-v2`; the converter continues to accept it.
