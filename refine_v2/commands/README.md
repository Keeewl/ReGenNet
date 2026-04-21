# refine_v2/commands

Minimal command set for refine_v2 module 1.

The scripts intentionally follow the same simple command-record style as
`refine/commands`: activate conda, set `CUDA_VISIBLE_DEVICES`, then run one
`python -m ...` command. Edit paths and numeric parameters directly in the
corresponding `.sh` file before running.

Files:

- `00_build_contact_labels.sh`: build `contact_labels_gt.npz` from `reaction_data`.
- `01_select_windows.sh`: build hand-time `selector_windows_v2_hand_time_tau010.npz` from coarse binary mesh contact.
- `02_audit_windows.sh`: strict + relaxed audit against direct GT contact labels.
- `03_vis_contact_labels.sh`: text inspection of one sample's GT contact labels.
- `04_vis_windows_vs_gt.sh`: text inspection of one sample's predicted windows vs GT.
- `05_run_minimal_loop.sh`: runs steps 00, 01, and 02.

Default inputs and outputs embedded in the scripts:

- `REACTION_DATA_PATH=refine/dataset/train/reaction_data.npz`
- `OUTPUT_DIR=refine_v2/outputs/train`
- `REGION_MAP_PATH=visualize/viewer/part_segm/6_parts/six_parts.pkl`
- selector relaxed params: `tau_contact=0.10`, `gap_merge=4`, `raw_L_min=2`, `window_size=30`
- caps stay unchanged: `per_hand_max_windows=2`, `per_seq_max_windows=3`
- current script device settings: `CUDA_VISIBLE_DEVICES=1`, `DEVICE=cuda`, `BATCH_SIZE=64`

Progress:

- `00_build_contact_labels.sh`, `01_select_windows.sh`, and `02_audit_windows.sh`
  print processed count, elapsed time, and ETA by default.
- Add `--no_progress` to the underlying `python -m ...` command if you want quiet output.

Examples:

```bash
bash refine_v2/commands/00_build_contact_labels.sh
bash refine_v2/commands/01_select_windows.sh
bash refine_v2/commands/02_audit_windows.sh

bash refine_v2/commands/03_vis_contact_labels.sh
bash refine_v2/commands/04_vis_windows_vs_gt.sh
```
