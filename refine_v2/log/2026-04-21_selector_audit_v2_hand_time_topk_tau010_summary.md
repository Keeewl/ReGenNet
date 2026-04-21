# refine_v2 Selector Audit V2 Hand-Time Top-K Tau010 Summary

Date: 2026-04-21

Artifact context:

- Module: `refine_v2` module 1
- GT label type: binary mesh-region contact labels
- Selector type: deterministic hand-time proposal selector with top-k region attribution
- Audit type: strict GT contact-label audit plus relaxed audit plus top-k audit
- Input space: restored pair space
- Dataset split/path: `refine/dataset/train/reaction_data.npz`
- GT labels: `refine_v2/outputs/train/contact_labels_gt.npz`
- Selector output: `refine_v2/outputs/train/selector_windows_v2_hand_time_topk_tau010.npz`
- Audit output: `refine_v2/outputs/train/selector_audit_v2_hand_time_topk_tau010.json`

## Selector Parameters

This run keeps GT labels, hand-time proposal, raw segment generation, and window rules unchanged.

- `proposal_type = hand_time_with_region_attribution`
- `selector_tau_contact = 0.10`
- `gap_merge = 4`
- `raw_L_min = 2`
- `window_size = 30`
- `per_hand_max_windows = 2`
- `per_seq_max_windows = 3`
- `top_k_regions = 3`
- `region_map_path = visualize/viewer/part_segm/6_parts/six_parts.pkl`

Important:

- GT contact labels remain from the existing GT pack.
- The selector threshold `0.10` does not change GT label definition.
- Proposal is generated per `sample x hand`.
- Region is assigned after proposal by attribution.
- Primary region is kept for strict audit compatibility.
- Top-k regions are added only as extra attribution, not as a new proposal axis.

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

Interpretation:

- Top-k attribution does not change the selector proposal count.
- Raw filtering is not the bottleneck.
- Cap pressure is acceptable and much lower than the previous hand-region proposal.
- Final average window count is about `2.10` per selected sequence, matching the intended small-window subset design.

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

Strict audit remains primary-region only. It is intentionally conservative and remains unchanged for compatibility.

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

Relaxed recall gives the time/hand proposal upper-bound view:

- Hand-only GT segment recall: `78.1%`
- Time-only GT segment recall: `83.7%`
- Time-only minus hand-only gap: `5.6` percentage points

## Top-K Audit Results

```text
topk_gt_segment_recall: 0.7285973526900086
topk_window_match_ratio: 0.7881031179237067
topk_region_match_ratio: 0.9745258050286723
top1_miss_topk_hit_count: 6580
top1_miss_topk_hit_ratio: 0.5640805829404201
top1_miss_topk_hit_ratio_over_all_gt: 0.35119555935098207
topk_recalled_gt_count: 13651
strict_missed_gt_count: 11665
```

Top-1 miss but top-k hit by GT region:

```text
torso_head: 1241
lower_body: 367
left_arm: 1312
right_arm: 1776
left_hand: 884
right_hand: 1000
```

Derived top-k audit numbers:

- Strict GT segment recall: `37.7%`
- Top-k GT segment recall: `72.9%`
- Hand-only GT segment recall: `78.1%`
- Time-only GT segment recall: `83.7%`
- Top-k minus strict: `35.1` percentage points
- Hand-only minus top-k: `5.2` percentage points
- Time-only minus hand-only: `5.6` percentage points
- Top-k window match ratio: `78.8%`
- Hand-only window match ratio: `79.0%`
- Top-k region match ratio: `97.5%`

Key interpretation:

```text
primary-region strict recall is low because single-primary region credit is too hard.
top-k attribution recovers most same-hand time-overlap region misses.
```

`top1_miss_topk_hit_ratio = 56.4%` means that among strict-missed GT segments, more than half are actually covered once the window is credited for its top-3 attributed regions.

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
- The global zero-window ratio of `39.9%` is mostly from GT-negative sequences, which is expected.
- `GT0 / Pred+ = 698`, or `16.4%` of GT-negative sequences, is the main sequence-level false-positive bucket to handle during subset construction.

## Comparison With Previous Hand-Time Primary-Only Run

The selector counts are unchanged:

```text
raw_segments_post_filter: 10869 -> 10869
windows_pre_cap: 14819 -> 14819
windows_post_cap: 11482 -> 11482
num_sequences_with_windows_post_cap: 5474 -> 5474
```

This confirms top-k only changes attribution/audit, not proposal.

Primary strict and relaxed metrics remain unchanged:

```text
gt_segment_recall: 0.3774
hand_only_gt_segment_recall: 0.7810
time_only_gt_segment_recall: 0.8368
window_contact_purity: 0.5762
false_positive_window_ratio: 0.2515
```

New top-k metrics add the missing diagnosis:

```text
topk_gt_segment_recall: 0.7286
topk_region_match_ratio: 0.9745
top1_miss_topk_hit_ratio: 0.5641
```

Conclusion:

```text
the selector/window timing is much better than primary-region strict recall suggests.
single-primary attribution is the dominant strict-recall bottleneck.
```

## Whether Selector / Window Is Good Enough

Judgment:

```text
The current selector/window is good enough to freeze as the module-1 proposal baseline.
```

Reasons:

1. GT-positive sequence coverage is high.

```text
gt_positive_zero_window_ratio = 1.57%
```

Only a small fraction of GT-positive sequences have no predicted window.

2. Time and hand proposal are adequate.

```text
time_only_gt_segment_recall = 83.7%
hand_only_gt_segment_recall = 78.1%
```

The proposal is not failing primarily at the time level.

3. Top-k region attribution nearly closes the region gap.

```text
topk_gt_segment_recall = 72.9%
topk_region_match_ratio = 97.5%
```

Top-k recall is only `5.2` percentage points below hand-only recall.

4. Cap pressure is no longer the dominant issue.

```text
cap_drop_ratio = 22.5%
```

Caps no longer create additional zero-window sequences.

5. Window count is practical for downstream subset/refiner.

```text
num_pred_windows = 11482
post-cap windows per selected sequence = 2.10
```

The output is compact enough for a focused contact subset.

Remaining caveats:

- `false_positive_window_ratio = 25.2%` is not negligible.
- `gt_negative_nonzero_window_ratio = 16.4%` means some GT-negative sequences produce windows.
- These should be handled during subset construction rather than by major selector redesign.

## Recommended Fixed Selector / Window Configuration

Use this as the module-1 selector/window baseline:

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

Artifact fields to keep:

- `primary_target_region`
- `primary_target_region_id`
- `topk_target_regions`
- `topk_target_region_ids`
- `topk_region_scores`
- `hand_contact_frame_ratio`
- `primary_region_contact_frame_ratio`
- `region_score_table`

Audit fields to keep:

- strict primary-region metrics
- relaxed hand/time metrics
- GT-positive / GT-negative split
- top-k metrics
- `top1_miss_topk_hit_*`
- wrong-region confusion

## Next Step

Proceed to contact subset construction.

Recommended subset policy:

1. Main positive subset:

```text
use GT+ / Pred+ sequences and their selected windows
```

2. Keep GT-negative predicted windows as a separate diagnostic or hard-negative bucket:

```text
GT0 / Pred+ = 698 sequences
```

Do not mix them silently into the main positive subset.

3. On the subset, rerun the same fixed selector/window configuration if a clean final training pack is needed.

4. Then implement the refiner data interface:

- subset manifest
- window crop loader
- window-level metadata
- top-k region annotation
- GT supervision alignment checks

5. Only after that, implement feature/network/loss/refiner.

Final decision:

```text
selector/window can be considered basically fixed for module 1.
the next productive work is subset + refiner data preparation, not further selector redesign.
```
