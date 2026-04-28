# 2026-04-28 Single Stage1 Infer/Export

Added a single-sample Stage1 inference/export tool for strict three-way visualization:

- `GT`
- `ReGenNet` (`cmdm` Stage1 only)
- `HiReact` (`cnetv5` Stage1 + Stage2 refine)

## Tool

- `sample/infer_single_stage1_clip.py`

## Purpose

Given one `dataset_key`, run one-sample Stage1 generation and export:

- `results.npy`
- `results_meta.npz`
- `map.txt`
- viewer-ready clip:
  - `<output_dir>/motions/<dataset_key>/P1.npz`
  - `<output_dir>/motions/<dataset_key>/P2.npz`

The exported clip can then be:

1. visualized directly as the Stage1 baseline;
2. fed into the existing Stage2 single-clip refiner to obtain HiReact.

## Commands

- CMDM baseline:
  - `refine_v2/commands/visual/27_infer_single_stage1_cmdm_by_dataset_key.sh`
- CNetV5 baseline:
  - `refine_v2/commands/visual/28_infer_single_stage1_cnetv5_by_dataset_key.sh`

## Notes

- `model_path` is intentionally left as an environment-variable placeholder in the command scripts, because the exact server-side checkpoint paths may differ from local files.
- The exported viewer clip is named by `dataset_key`, not by an index/action alias. This keeps GT / Stage1 / Stage2 alignment explicit.
