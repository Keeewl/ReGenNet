# refine_v2/commands

Minimal command set for refine_v2 module 1.

The scripts intentionally follow the same simple command-record style as
`refine/commands`: activate conda, set `CUDA_VISIBLE_DEVICES`, then run one
`python -m ...` command. Edit paths and numeric parameters directly in the
corresponding `.sh` file before running.

Files:

- `00_build_contact_labels.sh`: build `contact_labels_gt.npz` from `reaction_data`.
- `01_select_windows.sh`: build hand-time top-k `selector_windows_v2_hand_time_topk_tau010.npz` from coarse binary mesh contact.
- `02_audit_windows.sh`: strict + relaxed + top-k audit against direct GT contact labels.
- `03_vis_contact_labels.sh`: text inspection of one sample's GT contact labels.
- `04_vis_windows_vs_gt.sh`: text inspection of one sample's predicted windows vs GT.
- `05_run_minimal_loop.sh`: runs steps 00, 01, and 02.
- `06_build_action_type_stats.sh`: aggregate full-train contact/window stats by Inter-X action type.
- `07_build_subset_manifest.sh`: build a contact-rich sequence-level subset manifest.
- `08_rerun_selector_on_subset.sh`: rerun the frozen hand-time top-k selector on the main positive subset.
- `09_vis_subset_windows.sh`: print/export a text sanity report for subset windows.
- `10_view_subset_window_ait.sh`: open one chosen subset window interactively in aitviewer.
- `11_inspect_refiner_data.sh`: inspect the fast window-level refiner dataset and export a small JSON summary.
- `12_train_refiner_overfit.sh`: run the 64-window small overfit test for the first refiner.
- `12_train_refiner.sh`: train the first mesh-aware residual refiner on the contact-rich subset.
- `12_train_refiner_large_overnight.sh`: train a larger 8-layer/512-dim refiner for an overnight run.
- `13_eval_refiner.sh`: run window-level eval for a trained refine_v2 refiner checkpoint.

Default inputs and outputs embedded in the scripts:

- `REACTION_DATA_PATH=refine/dataset/train/reaction_data.npz`
- `OUTPUT_DIR=refine_v2/outputs/train`
- `REGION_MAP_PATH=visualize/viewer/part_segm/6_parts/six_parts.pkl`
- selector relaxed params: `tau_contact=0.10`, `gap_merge=4`, `raw_L_min=2`, `window_size=30`
- caps stay unchanged: `per_hand_max_windows=2`, `per_seq_max_windows=3`
- region attribution: `top_k_regions=3`
- current script device settings: `CUDA_VISIBLE_DEVICES=1`, `DEVICE=cuda`, `BATCH_SIZE=64`
- subset auto-selection defaults: `min_num_sequences=20`, `min_gt_positive_sequence_ratio=0.50`

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

bash refine_v2/commands/06_build_action_type_stats.sh
bash refine_v2/commands/07_build_subset_manifest.sh
bash refine_v2/commands/08_rerun_selector_on_subset.sh
bash refine_v2/commands/09_vis_subset_windows.sh
bash refine_v2/commands/10_view_subset_window_ait.sh
bash refine_v2/commands/11_inspect_refiner_data.sh
bash refine_v2/commands/12_train_refiner_overfit.sh
bash refine_v2/commands/12_train_refiner.sh
bash refine_v2/commands/12_train_refiner_large_overnight.sh
bash refine_v2/commands/13_eval_refiner.sh
```
