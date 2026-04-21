# Stage2 Refine Phase Goals

Created: 2026-04-21

Purpose:

This file records the current staged plan for implementing Stage2 refine in `refine_v2`.
It should be updated as each phase is completed, audited, or revised.

## Current Position

Module 1 is now implemented and audited:

- GT binary mesh-region contact labels
- restored pair space processing
- hand-time proposal selector
- top-k region attribution
- strict audit
- relaxed audit
- GT-positive / GT-negative sequence split
- top-k audit
- text inspection scripts

The latest top-k audit supports freezing the current selector/window baseline:

```text
proposal_type = hand_time_with_region_attribution
selector_tau_contact = 0.10
gap_merge = 4
raw_L_min = 2
window_size = 30
per_hand_max_windows = 2
per_seq_max_windows = 3
top_k_regions = 3
```

Key audit results:

```text
num_sequences = 9110
num_gt_positive_sequences = 4852
num_gt_negative_sequences = 4258
num_pred_windows = 11482
gt_segment_recall = 0.3774
hand_only_gt_segment_recall = 0.7810
time_only_gt_segment_recall = 0.8368
topk_gt_segment_recall = 0.7286
topk_window_match_ratio = 0.7881
topk_region_match_ratio = 0.9745
gt_positive_zero_window_ratio = 0.0157
gt_negative_nonzero_window_ratio = 0.1639
```

Interpretation:

- The proposal/window timing is adequate for module 1.
- Strict primary-region recall is too conservative.
- Top-k attribution recovers most of the primary-region misses.
- Selector/window should be treated as basically fixed for the next phase.
- The next productive step is not more selector redesign, but contact-rich subset construction.

## Main Strategy Update

The Stage2 training subset should not be selected only from selector outputs.

The preferred route is:

```text
full Inter-X train contact/action-type statistics
-> choose contact-rich action types
-> build sequence-level contact-rich subset
-> rerun fixed selector/window on the subset
-> audit subset windows
-> implement refiner data interface
-> implement refiner feature/network/loss/training
```

This is preferred because action type is a stable semantic sequence-level grouping,
while selector output can contain false positives.

Recommended division of responsibility:

```text
action type        -> sequence-level training-domain selection
GT contact labels  -> truth filtering and contact density measurement
selector/window    -> fixed window sampler inside the selected subset
top-k attribution  -> region annotation for downstream refiner supervision
```

## Phase 1: Full Train Contact Statistics By Action Type

Status: completed for first contact-rich subset pass

Goal:

Compute full Inter-X train statistics grouped by action type.

Required inputs:

- `refine/dataset/train/reaction_data.npz`
- `refine_v2/outputs/train/contact_labels_gt.npz`
- action type metadata from the existing dataset/reaction data pipeline
- optional selector/audit artifacts for window-level statistics:
  - `refine_v2/outputs/train/selector_windows_v2_hand_time_topk_tau010.npz`
  - `refine_v2/outputs/train/selector_audit_v2_hand_time_topk_tau010.json`

Required action-type metrics:

- `action_type`
- `num_sequences`
- `num_gt_positive_sequences`
- `gt_positive_sequence_ratio`
- `num_gt_segments`
- `gt_segments_per_sequence`
- `total_gt_contact_frames`
- `gt_contact_frame_ratio`
- `avg_gt_segment_length`
- `median_gt_segment_length`
- `num_selector_windows`
- `windows_per_sequence`
- `topk_gt_segment_recall`
- `topk_window_match_ratio`
- `window_contact_purity`
- `false_positive_window_ratio`

Useful derived scores:

```text
contact_rich_score =
  gt_positive_sequence_ratio
  * log(1 + gt_segments_per_sequence)
  * gt_contact_frame_ratio
```

Alternative training-oriented score:

```text
training_value_score =
  gt_contact_frame_ratio
  * windows_per_sequence
  * topk_window_match_ratio
```

Selection guardrails:

- Require enough samples per action type.
- Avoid selecting only tiny high-contact classes.
- Prefer action types with both high contact density and usable selector quality.
- Keep the output interpretable as a table for manual review.

Expected output:

- action-type statistics table, likely json/csv/md
- ranked contact-rich action type candidates
- initial recommended contact-rich subset action types

Completed outputs:

- `refine_v2/outputs/train/action_type_stats/action_type_stats.json`
- `refine_v2/outputs/train/action_type_stats/action_type_stats.csv`
- `refine_v2/outputs/train/action_type_stats/action_type_stats.md`

First-pass result:

- 40 action types were analyzed.
- 30 action types passed the initial broad recommendation rule.
- The broad rule was considered too wide for the first refiner subset.
- A narrower 15-action contact-rich subset was selected manually from the ranked statistics.

## Phase 2: Contact-Rich Sequence Subset

Status: completed for first 15-action subset

Goal:

Build a sequence-level subset using contact-rich action types.

Recommended subset logic:

1. Select action types from Phase 1.
2. Within those action types, split sequences into:

```text
GT+ / Pred+
GT+ / Pred0
GT0 / Pred+
GT0 / Pred0
```

3. Main positive training subset should focus on:

```text
GT+ / Pred+
```

4. Keep `GT0 / Pred+` as a separate diagnostic or hard-negative bucket.

5. Do not silently mix GT-negative predicted windows into the positive subset.

Expected output:

- subset manifest
- selected action type list
- sequence ids / dataset row indices
- bucket labels
- summary statistics

Selected 15 action types:

```text
A028 Hand wrestling
A025 Carry on back
A001 Handshake
A009 Sit on leg
A021 Dance
A000 Hug
A008 Pull
A019 Support with hand
A023 Shoulder to shoulder
A035 Help up
A027 Massaging leg
A022 Link arms
A003 Grab
A016 High-five
A034 Kiss on cheek
```

Completed outputs:

- `refine_v2/outputs/train/contact_subset/subset_manifest.json`
- `refine_v2/outputs/train/contact_subset/subset_sequences.csv`
- `refine_v2/outputs/train/contact_subset/main_positive_sequences.csv`
- `refine_v2/outputs/train/contact_subset/hard_negative_sequences.csv`
- `refine_v2/outputs/train/contact_subset/subset_summary.md`

## Phase 3: Rerun Fixed Selector/Window On Subset

Status: completed for first 15-action subset

Goal:

Rerun the frozen selector/window configuration on the selected subset.

Fixed selector/window configuration:

```text
proposal_type = hand_time_with_region_attribution
selector_tau_contact = 0.10
gap_merge = 4
raw_L_min = 2
window_size = 30
per_hand_max_windows = 2
per_seq_max_windows = 3
top_k_regions = 3
```

Required audit checks on subset:

- `gt_positive_zero_window_ratio`
- `topk_gt_segment_recall`
- `topk_window_match_ratio`
- `topk_region_match_ratio`
- `window_contact_purity`
- `false_positive_window_ratio`
- `gt_negative_nonzero_window_ratio`

Expected output:

- subset selector window artifact
- subset audit json
- subset audit log summary

Completed outputs:

- `refine_v2/outputs/train/contact_subset/selector_rerun/subset_selector_windows.npz`
- `refine_v2/outputs/train/contact_subset/selector_rerun/subset_selector_audit.json`
- `refine_v2/outputs/train/contact_subset/selector_rerun/subset_selector_audit_summary.md`
- `refine_v2/outputs/train/contact_subset/selector_rerun/subset_window_metadata.json`
- `refine_v2/outputs/train/contact_subset/selector_rerun/subset_window_metadata.csv`

Key subset audit results:

```text
num_sequences = 2842
num_gt_segments = 13190
num_pred_windows = 6749
gt_positive_zero_window_ratio = 0.0
topk_gt_segment_recall = 0.6860
topk_window_match_ratio = 0.8947
topk_region_match_ratio = 0.9655
window_contact_purity = 0.6857
false_positive_window_ratio = 0.1556
gt_negative_nonzero_window_ratio = 0.0
```

Decision:

```text
The 15-action contact-rich subset and fixed selector/window rerun are good enough
to become the first Stage2 refiner training domain.
```

## Phase 4: Refiner Data Interface

Status: completed for first fast-path implementation

Goal:

Prepare the data interface needed before implementing the network.

Required components:

- subset manifest loader
- selector window pack loader
- fixed-window crop loader
- restored-space consistency checks
- top-k region annotation loader
- GT supervision alignment checks
- per-window metadata export

Important design decision:

Do not duplicate one hand-time window into multiple region windows by default.
Keep one window with:

- primary region
- top-k regions
- region scores
- hand side
- time bounds
- contact labels

Expected output:

- minimal refiner dataset class
- data inspection CLI
- sanity-check commands

Completed outputs:

- `refine_v2/refiner_data/__init__.py`
- `refine_v2/refiner_data/schema.py`
- `refine_v2/refiner_data/sanity_checks.py`
- `refine_v2/refiner_data/feature_pack.py`
- `refine_v2/refiner_data/window_dataset.py`
- `refine_v2/refiner_data/window_loader.py`
- `refine_v2/refiner_data/README.md`
- `refine_v2/tools/inspect_refiner_data.py`
- `refine_v2/cli_inspect_refiner_data.py`
- `refine_v2/commands/11_inspect_refiner_data.sh`

Implemented sample unit:

```text
one sample = one hand-time selector window
```

Implemented fast-path fields:

```text
actor_motion_window
coarse_motion_window
gt_motion_window
coarse_region_contact_mask_window
coarse_min_region_dist_window
gt_region_contact_mask_window
gt_min_region_dist_window
hand_side / primary region / top-k region metadata
valid_mask
sequence and window metadata
```

Alignment policy:

```text
reaction_data row index = dataset_row_index
label_row_to_index      = {dataset_row_index -> label array index}
selector_row_to_index   = {dataset_row_index -> selector artifact local index}
manifest_row_to_record  = {dataset_row_index -> manifest sequence metadata}
```

Decision:

```text
The fast-path refiner data interface is good enough for the first trainable
refiner implementation.
```

Deferred:

```text
include_xyz=True remains NotImplementedError.
Dynamic SMPL-X xyz debug should be added only after the fast motion/contact
dataset and first refiner are stable.
```

## Phase 5: Refiner Feature / Network / Loss

Status: next

Goal:

Implement the first trainable refiner after subset/data loader are stable.

This phase is intentionally deferred until subset quality is audited.

Next first-scope requirements:

- model input packing from `RefineV2WindowDataset`
- minimal residual refiner architecture
- residual output over `coarse_motion_window`
- supervised target from `gt_motion_window`
- contact-aware auxiliary losses from GT mesh-region labels
- valid-mask-aware loss computation
- train/eval split protocol on the 15-action subset
- minimal checkpointing and metric logging
- one-batch and small-overfit tests before full subset training

Open design questions:

- whether top-k regions are used as conditioning, supervision candidates, or both
- whether to train on one primary region target or multi-region contact targets
- whether to include hard-negative windows from `GT0 / Pred+`
- whether refiner predicts full pose deltas, hand deltas, or contact-region corrections
- whether motion loss should initially be full-body MSE or weighted toward reactor hands/contact frames

## Current Recommendation

Freeze selector/window, the first 15-action subset, and the fast-path refiner
data interface for now.

Next concrete task:

```text
minimal residual refiner + train loop + loss + eval framework
```

The first refiner should be developed on the subset rerun outputs rather than on full train.

## Update Log

- 2026-04-21:
  - Module 1 selector/window judged basically fixed after top-k audit.
  - Main next phase changed to action-type contact-rich subset selection.
  - This `phase_goals` file created as the living plan for Stage2 refine.
  - Implemented action-type stats, contact-rich subset manifest, and subset selector rerun CLIs.
  - Full train action-type stats were run and used to select a 15-action contact-rich subset.
  - The 15-action subset selector rerun passed the current quality bar:
    - `topk_window_match_ratio = 0.8947`
    - `window_contact_purity = 0.6857`
    - `false_positive_window_ratio = 0.1556`
    - `gt_positive_zero_window_ratio = 0.0`
  - Next phase is subset visual sanity check plus refiner data/feature interface.
  - Added subset window text sanity inspection and aitviewer single-window inspection support.
  - Implemented fast-path `RefineV2WindowDataset`, feature packing, strict alignment checks, DataLoader collate, and inspection CLI.
  - Phase 4 is complete enough to move to the first trainable refiner framework.
