# 2026-04-26 Stage1 Clip -> Stage2 Exp8 Inference Bridge

Added a single-clip inference path that runs `refine_v2` directly on a viewer-ready Stage1 clip and exports a viewer-ready refined clip.

## Purpose

Support strict one-to-one comparison for:

- Stage1 snapshot clip
- the same clip after Stage2 `refiner_v2_exp8_interaction_v1_10k`

This avoids relying on the sampled `full_sequence_eval` pack, which may not contain the exact Stage1 clip of interest.

## Implementation

- Tool: `refine_v2/tools/infer_refiner_on_viewer_clip.py`
- CLI: `refine_v2/cli_infer_refiner_on_viewer_clip.py`

Pipeline:

1. Load viewer-ready clip (`P1.npz`, `P2.npz`)
2. Resolve actor/reactor roles
3. Convert SMPL-X params to Stage2 motion tensor
4. Run selector on the single clip
5. Build geometry features online for the selected windows
6. Run Stage2 refiner (`exp8`)
7. Stitch window residuals back to full sequence
8. Export refined viewer-ready clip

## Expected runtime

This path only runs inference on one clip, not training and not dataset-wide evaluation. Runtime should therefore be short enough to run once on the server and then download the output clip for local visualization.

## Commands

- Server-side inference:
  - `refine_v2/commands/visual/25_infer_refiner_on_stage1_clip_exp8.sh`
- Local viewing:
  - `refine_v2/commands/visual/26_view_snapshot_refined_stage1_clip_exp8.sh`
