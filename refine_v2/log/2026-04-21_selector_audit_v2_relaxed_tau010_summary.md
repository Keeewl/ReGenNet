# refine_v2 Selector Audit V2 Relaxed Tau010 Summary

Date: 2026-04-21

Artifact context:

- Module: `refine_v2` module 1
- GT label type: binary mesh-region contact labels
- Selector type: deterministic binary-contact selector v2
- Audit type: strict GT contact-label audit plus relaxed audit
- Input space: restored pair space
- Dataset split/path: `refine/dataset/train/reaction_data.npz`
- GT labels: `refine_v2/outputs/train/contact_labels_gt.npz`
- Selector output: `refine_v2/outputs/train/selector_windows_v2_relaxed_tau010.npz`
- Audit output: `refine_v2/outputs/train/selector_audit_v2_relaxed_tau010.json`

## Selector Parameters

This run keeps GT labels unchanged and only relaxes selector-side parameters:

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
- Window caps remain unchanged for this run.

## Selector Layered Statistics

```text
num_sequences: 9110
num_sequences_with_pred_contact_frames: 5539
num_pred_contact_frames_total: 948355
num_pred_contact_frames_per_sequence_mean: 104.10043907793633
num_raw_segments_pre_filter: 30657
num_raw_segments_post_filter: 29403
num_sequences_with_raw_segments_pre_filter: 5539
num_sequences_with_raw_segments_post_filter: 5473
num_windows_pre_cap: 36807
num_windows_post_cap: 12824
num_sequences_with_windows_pre_cap: 5473
num_sequences_with_windows_post_cap: 5473
num_windows_dropped_by_hand_cap: 21911
num_windows_dropped_by_seq_cap: 2072
avg_raw_segment_length_pre_filter: 31.192876015265682
avg_raw_segment_length_post_filter: 32.480563207835935
```

Derived selector numbers:

- Sequences with pred contact frames: `5539 / 9110 = 60.8%`
- Sequences with post-filter raw segments: `5473 / 9110 = 60.1%`
- Sequences with final windows: `5473 / 9110 = 60.1%`
- Zero-window sequences: `9110 - 5473 = 3637`
- Zero-window sequence ratio: `3637 / 9110 = 39.9%`
- Raw segments dropped by `raw_L_min`: `30657 - 29403 = 1254`
- Raw segment drop ratio: `1254 / 30657 = 4.1%`
- Sequences dropped by raw segment filter: `5539 - 5473 = 66`
- Windows dropped by caps: `36807 - 12824 = 23983`
- Window cap drop ratio: `23983 / 36807 = 65.2%`
- Windows dropped by hand cap: `21911 / 36807 = 59.5%`
- Windows dropped by seq cap: `2072 / 36807 = 5.6%`
- Pre-cap windows per nonzero sequence: `36807 / 5473 = 6.73`
- Post-cap windows per nonzero sequence: `12824 / 5473 = 2.34`

## Strict Audit Results

```text
num_sequences: 9110
num_gt_segments: 18736
num_pred_windows: 12824
gt_segment_recall: 0.42687873612297184
gt_contact_frame_coverage: 0.3176433001490424
avg_center_distance: 10.211483253588517
zero_window_sequence_ratio: 0.399231613611416
window_contact_purity: 0.5552635683094198
window_region_match_ratio: 0.6252333693622875
false_positive_window_ratio: 0.2666094822208359
matched_window_ratio: 0.7333905177791641
total_gt_contact_frames: 644783
covered_gt_contact_frames: 204811
```

## Relaxed Audit Results

```text
hand_only_gt_segment_recall: 0.7538428693424424
hand_only_window_match_ratio: 0.7935901434809731
time_only_gt_segment_recall: 0.8113791631084543
time_only_window_match_ratio: 0.8264971927635683
same_hand_time_overlap_window_count: 10177
same_hand_time_overlap_but_wrong_region_count: 3814
same_hand_time_overlap_but_wrong_region_ratio: 0.3747666306377125
same_sample_time_overlap_window_count: 10599
same_sample_time_overlap_but_wrong_hand_count: 2491
same_sample_time_overlap_but_wrong_hand_ratio: 0.23502217190300972
```

Derived relaxed audit numbers:

- Strict GT segment recall: `42.7%`
- Hand-only GT segment recall: `75.4%`
- Time-only GT segment recall: `81.1%`
- Hand-only minus strict recall: `32.7` percentage points
- Time-only minus hand-only recall: `5.8` percentage points
- Same-hand time-overlap wrong-region ratio: `37.5%`
- Same-sample time-overlap wrong-hand ratio: `23.5%`

## Comparison With Previous Default Selector

Previous default selector:

```text
tau_contact = 0.05
gap_merge = 2
raw_L_min = 4
window_size = 30
per_hand_max_windows = 2
per_seq_max_windows = 3
```

Main metric changes:

```text
num_pred_windows: 9921 -> 12824
zero_window_sequence_ratio: 0.5184 -> 0.3992
gt_segment_recall: 0.3969 -> 0.4269
gt_contact_frame_coverage: 0.3120 -> 0.3176
window_contact_purity: 0.7094 -> 0.5553
false_positive_window_ratio: 0.1071 -> 0.2666
```

Relaxing selector parameters increases windows and reduces zero-window sequences. However, strict recall and frame coverage only improve slightly, while purity drops and false positives increase substantially.

## Layer Diagnosis

### 1. Coarse contact mask layer

The relaxed selector produces many total pred contact frames:

```text
num_pred_contact_frames_total = 948355
GT contact frames = 644783
```

But only:

```text
num_sequences_with_pred_contact_frames = 5539 / 9110 = 60.8%
```

This means pred contact frames are unevenly distributed. Some sequences have many coarse contact frames, while about `3571` sequences have no coarse contact frame at all.

Conclusion:

```text
coarse contact mask still misses many sequences entirely.
```

### 2. Raw segment filter layer

```text
num_raw_segments_pre_filter = 30657
num_raw_segments_post_filter = 29403
```

Only `4.1%` of raw segments are removed by the relaxed `raw_L_min=2` filter.

Sequence-level loss is also small:

```text
5539 pre-filter sequences -> 5473 post-filter sequences
```

Only `66` sequences are lost at this layer.

Conclusion:

```text
raw segment filtering is not the main bottleneck in this relaxed run.
```

### 3. Window cap layer

```text
num_windows_pre_cap = 36807
num_windows_post_cap = 12824
```

Caps remove `23983` windows, about `65.2%` of all pre-cap candidates.

Most cap loss comes from `per_hand_max_windows`:

```text
num_windows_dropped_by_hand_cap = 21911
num_windows_dropped_by_seq_cap = 2072
```

However:

```text
num_sequences_with_windows_pre_cap = 5473
num_sequences_with_windows_post_cap = 5473
```

Caps do not create additional zero-window sequences in this run. They mainly reduce coverage inside sequences that already have candidates.

Conclusion:

```text
window caps are strong and likely limit contact-frame coverage,
but they are not the main cause of zero-window sequences.
```

### 4. Region assignment layer

Strict recall is much lower than hand-only relaxed recall:

```text
strict gt_segment_recall = 0.4269
hand_only_gt_segment_recall = 0.7538
```

The gap is `32.7` percentage points.

Also:

```text
same_hand_time_overlap_but_wrong_region_ratio = 0.3748
```

About `37.5%` of same-sample + same-hand + time-overlapping windows have a best-overlap GT region different from the predicted region.

Conclusion:

```text
region mismatch is a major strict-audit failure mode.
```

### 5. Hand assignment layer

Time-only recall is higher than hand-only recall:

```text
time_only_gt_segment_recall = 0.8114
hand_only_gt_segment_recall = 0.7538
```

The gap is `5.8` percentage points.

Also:

```text
same_sample_time_overlap_but_wrong_hand_ratio = 0.2350
```

Hand mismatch exists, but it is smaller than region mismatch as a recall bottleneck.

Conclusion:

```text
hand mismatch exists but is secondary to region mismatch.
```

## Main Conclusion

This relaxed run shows that the selector is no longer primarily limited by raw segment filtering.

The dominant issues are:

1. Region assignment mismatch.
2. Coarse contact mask has no response for many sequences.
3. Window caps are very strong and likely limit coverage depth, but do not cause zero-window sequences.

In short:

```text
The selector often finds the right time and hand,
but fails strict audit because the target region is different.
For sequences with no candidates, the problem happens before raw segments,
at the coarse contact mask layer.
```

## Recommended Next Steps

1. Add a wrong-region confusion matrix.
   - Count `pred_region -> best_same_hand_GT_region`.
   - This will show whether errors are concentrated in neighboring regions or caused by one dominant region.

2. Keep this relaxed selector output and test looser caps:
   - `per_hand_max_windows = 3`
   - `per_seq_max_windows = 5`

   This tests whether frame coverage improves when keeping more already-generated candidates.

3. Do not further relax `raw_L_min` first.
   - `raw_L_min=2` already shows raw filtering is not the bottleneck.

4. For zero-pred-contact sequences, inspect whether GT has contact.
   - If GT has contact but coarse has none, selector needs a fallback candidate mechanism beyond strict coarse contact mask.

5. Do not simply increase `tau_contact` further without diagnostics.
   - `tau=0.10` already increases false positives and reduces purity substantially.

