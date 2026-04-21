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

Status: implemented, waiting for full train run

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

## Phase 2: Contact-Rich Sequence Subset

Status: implemented, waiting for action-type stats results

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

## Phase 3: Rerun Fixed Selector/Window On Subset

Status: implemented, waiting for subset manifest results

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

## Phase 4: Refiner Data Interface

Status: pending

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

## Phase 5: Refiner Feature / Network / Loss

Status: pending

Goal:

Implement the first trainable refiner after subset/data loader are stable.

This phase is intentionally deferred until subset quality is audited.

Possible first-scope requirements:

- window-level input features
- coarse motion crop
- actor motion context
- hand side / top-k region annotations
- binary contact supervision
- restoration-aware output handling
- minimal train/eval loop

Open design questions:

- whether top-k regions are used as conditioning, supervision candidates, or both
- whether to train on one primary region target or multi-region contact targets
- whether to include hard-negative windows from `GT0 / Pred+`
- whether refiner predicts full pose deltas, hand deltas, or contact-region corrections

## Current Recommendation

Freeze selector/window for now.

Next concrete task:

```text
implement full train action-type contact statistics
```

Then decide the contact-rich subset from measured action-type statistics, not from intuition alone.

## Update Log

- 2026-04-21:
  - Module 1 selector/window judged basically fixed after top-k audit.
  - Main next phase changed to action-type contact-rich subset selection.
  - This `phase_goals` file created as the living plan for Stage2 refine.
  - Implemented action-type stats, contact-rich subset manifest, and subset selector rerun CLIs.
  - Full train action-type stats and subset rerun still need to be executed in the `regennet5090` environment.
