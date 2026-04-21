# refine_v2 Selector Audit V2 Summary

Date: 2026-04-21

Artifact context:

- Module: `refine_v2` module 1
- GT label type: binary mesh-region contact labels
- Selector type: deterministic binary-contact selector v2
- Audit type: strict GT contact-label audit
- Input space: restored pair space
- Dataset split/path assumed from current commands: `refine/dataset/train/reaction_data.npz`

## Core Parameters

These are the current default/core parameters used by `refine_v2/commands`:

- `tau_contact = 0.05`
- `gap_merge = 2`
- `raw_L_min = 4`
- `window_size = 30`
- `per_hand_max_windows = 2`
- `per_seq_max_windows = 3`
- `region_map_path = visualize/viewer/part_segm/6_parts/six_parts.pkl`
- target regions:
  - `torso_head`
  - `lower_body`
  - `left_arm`
  - `right_arm`
  - `left_hand`
  - `right_hand`

## Raw Audit Results

```text
num_sequences: 9110
num_gt_segments: 18736
num_pred_windows: 9921
gt_segment_recall: 0.39688300597779674
gt_contact_frame_coverage: 0.31200264274957623
avg_center_distance: 10.066832242041093
zero_window_sequence_ratio: 0.5184412733260154
window_contact_purity: 0.7093740550347747
window_region_match_ratio: 0.6518863833477884
false_positive_window_ratio: 0.10714645701038202
matched_window_ratio: 0.8928535429896179
total_gt_contact_frames: 644783
covered_gt_contact_frames: 201174
```

## Derived Numbers

- Average GT segments per sequence: `18736 / 9110 = 2.06`
- Average predicted windows per sequence: `9921 / 9110 = 1.09`
- Approximate zero-window sequences: `9110 * 0.5184412733260154 = 4722`
- Missed GT contact frames: `644783 - 201174 = 443609`
- Missed GT contact frame ratio: `1 - 0.31200264274957623 = 0.688`
- Approximate false-positive windows: `9921 * 0.10714645701038202 = 1063`
- Approximate matched windows: `9921 * 0.8928535429896179 = 8858`

## Summary

The current selector is high-precision but low-recall.

Once a predicted window is selected, it is usually meaningful:

- `matched_window_ratio = 0.893`
- `false_positive_window_ratio = 0.107`
- `window_contact_purity = 0.709`

This means about 89.3% of predicted windows overlap some GT contact, and about 70.9% of frames inside predicted windows are true GT contact frames.

The main failure mode is missing contact windows:

- `gt_segment_recall = 0.397`
- `gt_contact_frame_coverage = 0.312`
- `zero_window_sequence_ratio = 0.518`

About 51.8% of sequences produce no predicted windows at all. This is the biggest problem, because any GT contact inside those sequences cannot be covered. Overall, only 31.2% of GT contact frames are covered, leaving about 443,609 GT contact frames uncovered.

The region semantics are usable but not yet strong:

- `window_region_match_ratio = 0.652`

This suggests that time localization is often reasonable when a window exists, but about one third of same-hand overlaps point to a different region than the GT segment.

The center alignment is moderate:

- `avg_center_distance = 10.07` frames

With `window_size = 30`, this is not catastrophic, but it indicates that selected windows are not always centered on the GT segment.

## Interpretation

Current behavior:

```text
selected windows are usually contact-relevant,
but selector coverage is too conservative.
```

The selector is not primarily producing many false positives. It is more likely missing coarse contact due to strict distance thresholding, raw segment filtering, or final window caps.

## Recommended Next Checks

1. Check pred raw segment count before final window clipping.
   - If pred raw segments are already sparse, `tau_contact=0.05` is probably too strict for coarse motion.
   - If pred raw segments are sufficient but final windows are sparse, clipping by `per_hand_max_windows` or `per_seq_max_windows` is too aggressive.

2. Keep GT labels fixed at `tau_contact=0.05`, but try looser selector thresholds:
   - `selector tau_contact = 0.07`
   - `selector tau_contact = 0.08`
   - `selector tau_contact = 0.10`

3. Try less aggressive raw segment filtering:
   - `raw_L_min = 3`
   - `raw_L_min = 2`
   - `gap_merge = 4`

4. If pred raw segments are clipped heavily, try larger window caps:
   - `per_hand_max_windows = 3`
   - `per_seq_max_windows = 5`

5. Monitor these metrics in the next runs:
   - `zero_window_sequence_ratio`
   - `gt_segment_recall`
   - `gt_contact_frame_coverage`
   - `window_contact_purity`
   - `false_positive_window_ratio`
   - `window_region_match_ratio`

The immediate target should be reducing `zero_window_sequence_ratio` and increasing `gt_contact_frame_coverage`, while keeping `window_contact_purity` above roughly `0.55-0.65` and preventing `false_positive_window_ratio` from rising sharply.

