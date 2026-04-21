# refine_v2 Selector Audit V2 Hand-Time Tau010 Summary

Date: 2026-04-21

Artifact context:

- Module: `refine_v2` module 1
- GT label type: binary mesh-region contact labels
- Selector type: deterministic hand-time proposal selector with region attribution
- Audit type: strict GT contact-label audit plus relaxed audit
- Input space: restored pair space
- Dataset split/path: `refine/dataset/train/reaction_data.npz`
- GT labels: `refine_v2/outputs/train/contact_labels_gt.npz`
- Selector output: `refine_v2/outputs/train/selector_windows_v2_hand_time_tau010.npz`
- Audit output: `refine_v2/outputs/train/selector_audit_v2_hand_time_tau010.json`

## Selector Parameters

This run keeps GT labels unchanged and only changes selector-side proposal structure.

- `proposal_type = hand_time_with_region_attribution`
- `selector_tau_contact = 0.10`
- `gap_merge = 4`
- `raw_L_min = 2`
- `window_size = 30`
- `per_hand_max_windows = 2`
- `per_seq_max_windows = 3`
- `region_map_path = visualize/viewer/part_segm/6_parts/six_parts.pkl`

Important:

- GT contact labels remain from the existing GT pack.
- The selector threshold `0.10` does not change GT label definition.
- Proposal is generated per `sample x hand`, not per `sample x hand x region`.
- Region is assigned after proposal by attribution, using each segment/window's primary target region.

## Selector Layered Statistics

```text
num_sequences: 9110
num_sequences_with_pred_contact_frames: 5539
num_pred_contact_frames_total: 470059
num_pred_contact_frames_per_sequence_mean: 51.59813391877058
num_raw_segments_pre_filter: 11246
num_raw_segments_post_filter: 10869
num_sequences_with_raw_segments_pre_filter: 5539
num_sequences_with_raw_segments_post_filter: 5474
num_windows_pre_cap: 14819
num_windows_post_cap: 11482
num_sequences_with_windows_pre_cap: 5474
num_sequences_with_windows_post_cap: 5474
num_windows_dropped_by_hand_cap: 1949
num_windows_dropped_by_seq_cap: 1388
avg_raw_segment_length_pre_filter: 42.10830517517339
avg_raw_segment_length_post_filter: 43.53417977734842
```

Derived selector numbers:

- Sequences with pred contact frames: `5539 / 9110 = 60.8%`
- Sequences with post-filter raw segments: `5474 / 9110 = 60.1%`
- Sequences with final windows: `5474 / 9110 = 60.1%`
- Zero-window sequences: `9110 - 5474 = 3636`
- Zero-window sequence ratio: `3636 / 9110 = 39.9%`
- Raw segments dropped by `raw_L_min`: `11246 - 10869 = 377`
- Raw segment drop ratio: `377 / 11246 = 3.4%`
- Sequences dropped by raw segment filter: `5539 - 5474 = 65`
- Windows dropped by caps: `14819 - 11482 = 3337`
- Window cap drop ratio: `3337 / 14819 = 22.5%`
- Windows dropped by hand cap: `1949 / 14819 = 13.2%`
- Windows dropped by seq cap: `1388 / 14819 = 9.4%`
- Pre-cap windows per nonzero sequence: `14819 / 5474 = 2.71`
- Post-cap windows per nonzero sequence: `11482 / 5474 = 2.10`

## Strict Audit Results

```text
num_sequences: 9110
num_gt_segments: 18736
num_pred_windows: 11482
gt_segment_recall: 0.37740179333902646
gt_contact_frame_coverage: 0.294539403179054
avg_center_distance: 11.311496392832208
zero_window_sequence_ratio: 0.3991218441273326
window_contact_purity: 0.576168495616327
window_region_match_ratio: 0.6686149095721218
false_positive_window_ratio: 0.2515241247169483
matched_window_ratio: 0.7484758752830517
total_gt_contact_frames: 644783
covered_gt_contact_frames: 189914
```

## Relaxed Audit Results

```text
hand_only_gt_segment_recall: 0.7809564474807856
hand_only_window_match_ratio: 0.7897578819021076
time_only_gt_segment_recall: 0.8368381725021349
time_only_window_match_ratio: 0.8262497822678976
same_hand_time_overlap_window_count: 9068
same_hand_time_overlap_but_wrong_region_count: 3005
same_hand_time_overlap_but_wrong_region_ratio: 0.3313850904278782
same_sample_time_overlap_window_count: 9487
same_sample_time_overlap_but_wrong_hand_count: 2441
same_sample_time_overlap_but_wrong_hand_ratio: 0.25729946242226204
```

Derived relaxed audit numbers:

- Strict GT segment recall: `37.7%`
- Hand-only GT segment recall: `78.1%`
- Time-only GT segment recall: `83.7%`
- Hand-only minus strict recall: `40.4` percentage points
- Time-only minus hand-only recall: `5.6` percentage points
- Same-hand time-overlap wrong-region ratio: `33.1%`
- Same-sample time-overlap wrong-hand ratio: `25.7%`

## GT-Positive / GT-Negative Sequence Split

```text
num_gt_positive_sequences: 4852
num_gt_negative_sequences: 4258
gt_positive_zero_window_count: 76
gt_positive_zero_window_ratio: 0.015663643858202802
gt_negative_zero_window_count: 3560
gt_negative_zero_window_ratio: 0.8360732738374824
gt_negative_nonzero_window_count: 698
gt_negative_nonzero_window_ratio: 0.1639267261625176
gt_positive_pred_positive_count: 4776
gt_positive_pred_zero_count: 76
gt_negative_pred_positive_count: 698
gt_negative_pred_zero_count: 3560
```

Sequence buckets:

- `GT+ / Pred+`: `4776`
- `GT+ / Pred0`: `76`
- `GT0 / Pred+`: `698`
- `GT0 / Pred0`: `3560`

Interpretation:

- `GT+ / Pred0` is the true sequence-level missed-contact bucket.
- Only `76 / 4852 = 1.57%` of GT-positive sequences have no predicted window.
- The global `zero_window_sequence_ratio = 39.9%` is mostly from GT-negative sequences, which is expected.
- `GT0 / Pred+ = 698`, or `16.4%` of GT-negative sequences, is the main sequence-level false-positive bucket.

## Comparison With Previous Hand-Region Relaxed Tau010 Selector

Previous run:

- Selector output: `refine_v2/outputs/train/selector_windows_v2_relaxed_tau010.npz`
- Proposal type: `hand x region`
- Same selector-side loose parameters: `tau_contact=0.10`, `gap_merge=4`, `raw_L_min=2`, `window_size=30`, `per_hand_max_windows=2`, `per_seq_max_windows=3`

Main selector changes:

```text
raw_segments_post_filter: 29403 -> 10869
windows_pre_cap: 36807 -> 14819
windows_post_cap: 12824 -> 11482
num_windows_dropped_by_hand_cap: 21911 -> 1949
num_windows_dropped_by_seq_cap: 2072 -> 1388
num_sequences_with_windows_post_cap: 5473 -> 5474
```

The hand-time proposal removes most region-duplicated candidates. Cap pressure drops sharply:

- Previous cap drop ratio: `(36807 - 12824) / 36807 = 65.2%`
- Hand-time cap drop ratio: `(14819 - 11482) / 14819 = 22.5%`

Main audit changes:

```text
gt_segment_recall: 0.4269 -> 0.3774
gt_contact_frame_coverage: 0.3176 -> 0.2945
window_contact_purity: 0.5553 -> 0.5762
window_region_match_ratio: 0.6252 -> 0.6686
false_positive_window_ratio: 0.2666 -> 0.2515
hand_only_gt_segment_recall: 0.7538 -> 0.7810
time_only_gt_segment_recall: 0.8114 -> 0.8368
same_hand_time_overlap_but_wrong_region_ratio: 0.3748 -> 0.3314
```

The time proposal improves:

- Hand-only recall improves from `75.4%` to `78.1%`.
- Time-only recall improves from `81.1%` to `83.7%`.

The region assignment also improves:

- Region match ratio improves from `62.5%` to `66.9%`.
- Same-hand wrong-region ratio drops from `37.5%` to `33.1%`.

However, strict recall decreases:

- Strict GT segment recall drops from `42.7%` to `37.7%`.
- GT contact frame coverage drops from `31.8%` to `29.5%`.

This means the proposal timing is better, but strict `sample + hand + region` recall is hurt by representing each hand-time window with only one primary attributed region.

## Layer Diagnosis

### 1. Coarse / hand-mask layer

The number of sequences with predicted contact frames is unchanged from the previous relaxed run:

```text
num_sequences_with_pred_contact_frames = 5539 / 9110 = 60.8%
```

The total pred contact frames are lower than the previous hand-region run because hand-time counts a hand-level union instead of separately counting every hand-region contact frame.

Conclusion:

```text
hand-time aggregation reduces duplicate region counts; it does not reduce sequence-level coarse coverage.
```

### 2. Raw segment layer

```text
num_raw_segments_pre_filter = 11246
num_raw_segments_post_filter = 10869
```

Only `3.4%` of raw segments are removed by `raw_L_min=2`, and only `65` sequences are lost at this layer.

Conclusion:

```text
raw segment filtering is not the bottleneck.
```

### 3. Window cap layer

```text
num_windows_pre_cap = 14819
num_windows_post_cap = 11482
```

Caps remove `3337` windows, or `22.5%` of pre-cap candidates. This is much lower than the previous hand-region run.

Also:

```text
num_sequences_with_windows_pre_cap = 5474
num_sequences_with_windows_post_cap = 5474
```

Caps do not create additional zero-window sequences.

Conclusion:

```text
cap is no longer the main structural bottleneck, though it can still reduce within-sequence contact-frame coverage.
```

### 4. Region attribution / strict audit layer

The key gap is:

```text
strict gt_segment_recall = 37.7%
hand_only_gt_segment_recall = 78.1%
time_only_gt_segment_recall = 83.7%
```

Time and hand coverage are much higher than strict region-aware coverage. Region mismatch improves compared with the previous run, but strict recall still drops because a single hand-time window has only one primary target region.

Conclusion:

```text
the current main bottleneck is not time proposal, raw filtering, or zero-window sequences;
it is single-primary region attribution under strict region-aware evaluation.
```

## Final Summary

Hand-time proposal is useful and should be kept as the proposal structure:

- It removes most hand-region duplicated candidates.
- It greatly reduces cap pressure.
- It improves hand-only and time-only recall.
- It slightly improves purity and reduces false-positive window ratio.
- It makes zero-window interpretation clearer: only `1.57%` of GT-positive sequences have zero windows.

The strict recall drop is expected under the current single-primary-region artifact:

- A hand-time window may overlap GT contacts from multiple regions.
- Strict audit only credits the window for its primary attributed region.
- Therefore timing can be correct while strict region recall remains low.

Recommended next step:

1. Keep hand-time proposal.
2. Save `top_k_target_regions` per raw segment/window, such as top-2 or top-3 attributed regions.
3. Add audit metrics for `topk_region_match_ratio` and `topk_strict_recall`.
4. Keep primary-region strict audit as the conservative metric, but use top-k audit to decide whether the proposal is already adequate for downstream contact subset construction.
5. After top-k attribution is measured, tune `tau_contact` or window caps only if the top-k recall still fails to approach hand-only recall.
