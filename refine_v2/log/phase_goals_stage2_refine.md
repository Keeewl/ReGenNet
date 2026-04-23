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

## Phase 5: First Refiner Training Framework

Status: completed for first baseline

Goal:

Implement the first trainable refiner after subset/data loader are stable.

Completed baseline:

- `RefineV2WindowRefiner`
- mesh-aware condition encoder
- residual output over `coarse_motion_window`
- contact-weighted motion loss
- temporal residual smoothness loss
- sequence-level train/val split
- checkpoint / resume / logging
- window-level coarse-vs-refined eval
- small overfit test

Current result:

```text
64-window overfit passed.
Large exp2 best heldout val checkpoint at step 4000 improved over coarse.
Long 80k training overfit, so early stopping / regularization is needed.
```

Decision:

```text
The training scaffold is valid, but motion reconstruction eval is not sufficient
as the Stage2 refine success criterion.
```

## Phase 6: Contact-Centric Eval And Visualization

Status: next

Goal:

Add a Stage2-specific evaluation layer that directly measures whether the
refiner improves reactor hand physical/contact quality.

Rationale:

Stage2 refine is not only a motion-reconstruction task. The main target is:

```text
improve reactor hand motion physical/contact quality
```

Therefore, the main iteration loop should be:

```text
0. freeze selector/window/subset
1. update refiner feature/model/loss
2. evaluate with eval_contact + visualization
3. use contact diagnostics to update step 1
```

This avoids optimizing only `pred_motion_error` while missing the real contact
quality objective.

Recommended first `eval_contact` metrics:

- hand-region min-distance error:
  - `abs(pred_min_dist - gt_min_dist)`
  - compare against `abs(coarse_min_dist - gt_min_dist)`
- GT contact-frame distance improvement:
  - `coarse_dist - pred_dist`
- binary contact precision / recall / F1:
  - using the existing GT contact labels
  - start with `tau_contact = 0.05`
- top-k region contact improvement:
  - evaluate only window top-k regions as a focused diagnostic
- contact frequency / duration error:
  - predicted contact frame count
  - predicted contact segment duration
- contact jitter / flicker:
  - number of contact mask transitions

Recommended breakdowns:

- action type
- hand side
- primary region
- top-k region
- window vs full selected sequence

Visualization requirements:

- timeline view for one whole sequence:
  - selected windows
  - GT/coarse/refined contact masks
  - min-distance curves
  - hand side and top-k region labels
- single-window aitviewer inspection:
  - coarse vs refined vs GT
  - window start/end annotation
  - hand / primary / top-k region annotation

Important implementation rule:

```text
Implement contact eval and visualization first as offline evaluation.
Do not immediately put dynamic SMPL-X geometry forward into the training loop.
```

This keeps training fast while allowing direct contact-quality validation.

## Phase 7: Refiner Feature / Model / Loss Iteration

Status: pending after Phase 6 contact eval

Goal:

Use `eval_contact` and visualization diagnostics to improve the refiner.

Next likely feature/model/loss updates:

- add distance trend features:
  - `dist[t] - dist[t-1]`
- strengthen coarse min-distance/contact encoding
- add window-relative time embedding
- add selected-hand / contact-side joint weighting
- add residual magnitude regularization
- add non-contact preservation loss:
  - keep `pred` close to `coarse` outside GT contact frames
- consider hard-negative `GT0 / Pred+` only after positive subset behavior is stable
- add optional slow geometry loss only after offline `eval_contact` proves the metric is meaningful

Training recommendations:

- do not keep increasing model size blindly
- prefer shorter runs with early stopping
- use heldout sequence-level val as the model-selection signal
- use contact eval as the Stage2 success signal

Open design questions:

- whether top-k regions are used as conditioning, supervision candidates, or both
- whether refiner predicts full pose deltas, hand deltas, or contact-region corrections
- whether motion loss should initially be full-body or weighted toward reactor hands/contact frames
- which contact metric should be the primary model-selection criterion
- whether geometry loss should be enabled during training or kept as eval-only

## Current Recommendation

Freeze selector/window, the first 15-action subset, and the fast-path refiner
data interface for now.

Next concrete task:

```text
offline eval_contact_refiner + sequence/window contact visualization
```

Then use those contact metrics to update feature/model/loss.

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
  - Implemented the first trainable refine_v2 residual refiner framework.
  - A 64-window overfit test passed (`loss_total` decreased from `0.01390` to `0.00812`).
  - Large exp2 training validated the refiner direction but overfit after early steps:
    - best heldout val checkpoint at step 4000
    - best val `pred_motion_error = 0.01547` vs coarse `0.01652`
    - final step 80000 overfit and became worse than coarse on heldout val
  - Next training priority is shorter, better-regularized runs plus early stopping / patience.
- 2026-04-22:
  - Updated the phase plan around the real Stage2 objective:
    - freeze selector/window/subset
    - iterate feature/model/loss
    - validate with contact-centric metrics and visualization
  - Added Phase 6 as the next priority:
    - offline `eval_contact_refiner`
    - sequence/window contact timeline visualization
    - coarse/refined/GT contact-quality comparison
  - Added Phase 7 as the follow-up:
    - improve refiner features/losses based on `eval_contact` diagnostics
    - avoid blind model-size increases
    - add geometry loss only after offline contact eval is trustworthy
  - Boundary translation loss experiment:
    - exp3 used `lambda_boundary_trans=2.0`, `boundary_trans_frames=2`
    - motion eval improved slightly versus exp2, and window-local boundary
      transl jump stayed close to coarse
    - contact geometry improvement dropped clearly versus exp2:
      - exp3 `refined_contact_f1 = 0.8187`
      - exp3 `topk_refined_contact_f1 = 0.8264`
      - exp3 `gt_contact_contact_dist_improvement = 0.00236`
    - conclusion: boundary transl anchor is directionally useful, but
      `lambda_boundary_trans=2.0` is too conservative for contact improvement
  - Next run:
    - exp4 uses `lambda_boundary_trans=1.0`
    - keeps `boundary_trans_frames=2`
    - default training length is reduced to `num_steps=10000`
    - save path moves to `refine_v2/save/train/refiner_v2_exp4_boundary_lam1_10k`
  - Command organization rule from now on:
    - new training commands go under `refine_v2/commands/train/`
    - new eval commands go under `refine_v2/commands/eval/`
    - new visualization commands go under `refine_v2/commands/visual/`
    - new training/eval/visual experiment outputs go under `refine_v2/save/`
    - old flat `commands/*.sh` files are kept untouched for reproducibility
  - Stage1/Stage2 responsibility clarification:
    - Stage1 is responsible for producing the coarse reactor motion, including
      broad global translation quality
    - Stage2's target is contact refinement, mainly hand/arm/contact quality
      with controlled full-body residuals
    - large reactor translation errors are fundamentally a Stage1 issue
    - Stage1 is frozen for now, so Stage2 will keep using boundary/regularized
      constraints to avoid making transl discontinuity worse
    - if time allows later, Stage1 should be revisited for coarse transl and
      global interaction alignment quality
  - exp3 vs exp4 boundary baseline selection:
    - exp2 is no longer considered a good practical baseline because
      visualization showed obvious reactor translation discontinuity, even
      though its contact metrics are strongest
    - exp3:
      - `lambda_boundary_trans = 2.0`
      - `boundary_trans_frames = 2`
      - `all_valid_dist_l1_improvement = 0.00239`
      - `gt_contact_contact_dist_improvement = 0.00236`
      - `refined_contact_f1 = 0.8187`
      - `topk_refined_contact_f1 = 0.8264`
      - `surrogate_penetration_depth_improvement = -0.000052`
      - window-local boundary jump stayed close to coarse
    - exp4:
      - `lambda_boundary_trans = 1.0`
      - `boundary_trans_frames = 2`
      - `num_steps = 10000`
      - `all_valid_dist_l1_improvement = 0.00201`
      - `gt_contact_contact_dist_improvement = 0.00202`
      - `refined_contact_f1 = 0.8155`
      - `topk_refined_contact_f1 = 0.8232`
      - `surrogate_penetration_depth_improvement = -0.000049`
    - conclusion:
      - lowering boundary loss from `2.0` to `1.0` did not recover contact
        quality
      - exp4 is slightly safer on surrogate penetration but slightly worse on
        contact recall/F1/distance
      - among the boundary-constrained candidates, exp3 is the better current
        stable baseline because it gives stronger contact improvement while
        still controlling translation discontinuity
      - exp4 remains useful as a conservative reference, not as the main
        baseline
  - Next refiner design direction:
    - stop spending main effort on scalar boundary-loss tuning
    - move to model/loss scope control:
      - hand/arm residual should have high freedom
      - root/transl should have low freedom and boundary/continuity anchors
      - lower body should mostly preserve coarse motion
    - next implementation phase should focus on:
      - joint-group gated residual
      - group-weighted motion/preservation losses
      - window phase and boundary indicators as features
      - scope eval metrics: hand/arm/torso/lower-body/transl delta norms
      - full-sequence continuity eval after stitching
  - refine_v2_v1 scope-geometry implementation completed:
    - added offline relative geometry feature cache
    - added optional dataset/cache alignment validation
    - added geometry-conditioned per-frame condition encoder
    - added fixed joint-group gated residual scaling
    - added group-weighted motion loss and hand/arm contact-frame loss
    - added joint-group delta-norm eval metrics
    - added grouped commands under:
      - `refine_v2/commands/features/`
      - `refine_v2/commands/train/`
      - `refine_v2/commands/eval/`
      - `refine_v2/commands/visual/`
    - new exp5 paths:
      - feature cache: `refine_v2/save/features/scope_geom_train/geometry_feature_cache.npz`
      - train output: `refine_v2/save/train/refiner_v2_exp5_scope_geom_10k`
    - validation completed:
      - py_compile passed
      - all relevant CLI `--help` checks passed
      - geometry-enabled model/loss smoke test passed
      - legacy no-geometry model forward smoke test passed
  - Next concrete run:
    - build geometry cache
    - train `refiner_v2_exp5_scope_geom_10k`
    - compare against exp3 using window eval, contact eval, and aitviewer visual pack
- 2026-04-23:
  - exp5 geometry cache completed quickly and correctly:
    - `6749` windows
    - elapsed `00:00:22`
    - `primary_relative_vector_window = [6749, 3, 30]`
    - `primary_relative_dist_window = [6749, 30]`
    - `topk_relative_vectors_window = [6749, 3, 3, 30]`
    - `topk_relative_dists_window = [6749, 3, 30]`
    - speed is expected because this cache only computes sparse window-level
      centroid geometry, not dense region-to-region mesh min-distance.
  - exp5 window eval completed:
    - `pred_motion_error = 0.0130820452`
    - `motion_improvement = 0.0032817393`
    - `pred_contact_motion_error = 0.0130605551`
    - `contact_motion_improvement = 0.0034877443`
    - `boundary_trans_jump_excess = -0.0000008370`
    - `pred_boundary_trans_jump = 0.0097265707`
    - `coarse_boundary_trans_jump = 0.0097274078`
    - interpretation: exp5 improves window motion/contact-frame motion while
      keeping boundary translation essentially unchanged from coarse.
  - exp5 scope/delta eval confirms that the v1 design is doing the intended
    hand/arm-focused refinement:
    - `delta_norm_selected_hand = 0.0124185895`
    - `delta_norm_same_side_arm = 0.0155224866`
    - `delta_norm_other_hand_arm = 0.0110871044`
    - `delta_norm_lower_body = 0.0044759440`
    - `delta_norm_torso_root = 0.0029985403`
    - `delta_norm_transl = 0.0002474822`
    - selected hand delta is about `50x` transl delta
    - same-side arm delta is about `63x` transl delta
  - exp5 contact eval completed:
    - `all_valid_dist_l1_improvement = 0.0027992890`
    - `gt_contact_contact_dist_improvement = 0.0028254371`
    - `coarse_contact_f1 = 0.8003375103`
    - `refined_contact_f1 = 0.8221591739`
    - `topk_coarse_contact_f1 = 0.8083653870`
    - `topk_refined_contact_f1 = 0.8297871497`
    - `surrogate_penetration_depth_improvement = -0.0000661652`
  - exp5 vs exp3:
    - `all_valid_dist_l1_improvement` improves by about `17.1%`
    - `gt_contact_contact_dist_improvement` improves by about `19.6%`
    - `refined_contact_f1` improves by about `+0.0035`
    - `topk_refined_contact_f1` improves by about `+0.0034`
    - surrogate penetration is slightly worse than exp3
  - exp5 is now the current practical baseline:
    - better than exp3 on motion eval
    - better than exp3 on contact-distance eval
    - better than exp3 on refined/top-k contact F1
    - translation remains controlled
    - residual scope matches the Stage2 hand/arm contact-refinement target
  - exp5 is not yet the final upper-bound model:
    - `refined_contact_f1` is still below the target `0.84`
    - `topk_refined_contact_f1` is still below the target `0.845`
    - `gt_contact_contact_dist_improvement` is still below the target `0.004`
    - surrogate penetration is slightly worse and needs visual inspection
  - Breakdown observations:
    - strongest action-type gains: `High-five`, `Dance`, `Massaging leg`,
      `Hand wrestling`, `Sit on leg`, `Handshake`
    - weakest gains: `Pull`, `Support with hand`, `Hug`, `Link arms`, `Help up`
    - strongest primary-region gains are on `left_hand` and `right_hand`,
      which matches the Stage2 objective
  - Next concrete task:
    - export and inspect exp5 aitviewer visual packs
    - compare exp5 against exp3/exp2 visually on direct hand contact and weak
      action types
    - if visible over-close/penetration exists, add a light anti-overclose or
      anti-penetration regularizer
    - if weak action types remain poor, extend geometry features beyond simple
      selected-hand to target-region centroid features
