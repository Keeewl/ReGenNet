# refine_v2 Contact Subset 15 Selector Rerun Summary

Date: 2026-04-21

Artifact context:

- Module: `refine_v2` Stage2 subset pipeline
- Subset type: contact-rich action-type subset
- Number of selected action types: `15`
- Rerun bucket: `GT+ / Pred+`
- Selector type: frozen hand-time proposal with top-k region attribution
- Input space: restored pair space
- GT labels: `refine_v2/outputs/train/contact_labels_gt.npz`
- Subset manifest: `refine_v2/outputs/train/contact_subset/subset_manifest.json`
- Subset selector output: `refine_v2/outputs/train/contact_subset/selector_rerun/subset_selector_windows.npz`
- Subset audit output: `refine_v2/outputs/train/contact_subset/selector_rerun/subset_selector_audit.json`
- Subset window metadata: `refine_v2/outputs/train/contact_subset/selector_rerun/subset_window_metadata.json`

## Fixed Selector Configuration

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

This rerun keeps the selector/window protocol fixed. The rerun only changes the input domain to the selected `GT+ / Pred+` contact-rich subset.

## Selected Action Types

The first contact-rich subset is considered basically fixed to these 15 action types:

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

Rationale:

- These were selected from full train action-type contact statistics.
- They include the strongest `contact_rich_score` and `training_value_score` candidates.
- They cover hand-hand, hand-arm, hand-body, support, pull/help, and local face/cheek contact patterns.
- The subset is intentionally narrower than the initial 30 recommended candidates to keep it genuinely contact-rich.

## Rerun Command Context

```text
python -m refine_v2.cli_rerun_selector_on_subset
  --reaction_data_path refine/dataset/train/reaction_data.npz
  --contact_labels_path refine_v2/outputs/train/contact_labels_gt.npz
  --subset_manifest_path refine_v2/outputs/train/contact_subset/subset_manifest.json
  --output_dir refine_v2/outputs/train/contact_subset/selector_rerun
  --region_map_path visualize/viewer/part_segm/6_parts/six_parts.pkl
  --include_buckets "GT+ / Pred+"
  --tau_contact 0.10
  --gap_merge 4
  --raw_L_min 2
  --window_size 30
  --per_hand_max_windows 2
  --per_seq_max_windows 3
  --top_k_regions 3
  --batch_size 64
  --num_workers 4
  --device cuda
  --frame_chunk 1
  --target_chunk 2048
```

Runtime:

```text
select_windows: 2842 / 2842 samples, elapsed 00:01:48
audit_windows: 22781 / 22781 items, elapsed 00:00:01
```

## Subset Selector Audit Results

```text
num_sequences: 2842
num_gt_segments: 13190
num_pred_windows: 6749
gt_positive_zero_window_ratio: 0.0
topk_gt_segment_recall: 0.6859742228961334
topk_window_match_ratio: 0.8946510594162098
topk_region_match_ratio: 0.9654545454545455
window_contact_purity: 0.6856768904035165
false_positive_window_ratio: 0.15557860423766484
gt_negative_nonzero_window_ratio: 0.0
```

Derived values:

```text
gt_segments_per_sequence = 13190 / 2842 = 4.64
windows_per_sequence = 6749 / 2842 = 2.37
```

Interpretation:

- The subset contains dense contact supervision.
- Each sequence has multiple GT contact segments on average.
- Each sequence has a small number of selected windows, matching the intended local-refinement design.
- The main positive bucket has no sequence-level zero-window issue.

## Comparison With Full Train Top-K Run

Full train top-k run:

```text
num_sequences: 9110
num_gt_segments: 18736
num_pred_windows: 11482
topk_gt_segment_recall: 0.7286
topk_window_match_ratio: 0.7881
topk_region_match_ratio: 0.9745
window_contact_purity: 0.5762
false_positive_window_ratio: 0.2515
gt_positive_zero_window_ratio: 0.0157
gt_negative_nonzero_window_ratio: 0.1639
```

Subset rerun:

```text
num_sequences: 2842
num_gt_segments: 13190
num_pred_windows: 6749
topk_gt_segment_recall: 0.6860
topk_window_match_ratio: 0.8947
topk_region_match_ratio: 0.9655
window_contact_purity: 0.6857
false_positive_window_ratio: 0.1556
gt_positive_zero_window_ratio: 0.0
gt_negative_nonzero_window_ratio: 0.0
```

Main changes:

```text
topk_window_match_ratio: 0.7881 -> 0.8947
window_contact_purity: 0.5762 -> 0.6857
false_positive_window_ratio: 0.2515 -> 0.1556
gt_positive_zero_window_ratio: 0.0157 -> 0.0
gt_negative_nonzero_window_ratio: 0.1639 -> 0.0
topk_region_match_ratio: 0.9745 -> 0.9655
topk_gt_segment_recall: 0.7286 -> 0.6860
```

Interpretation:

- Window quality improves substantially in the contact-rich subset.
- False-positive windows drop clearly.
- Contact purity improves by about `10.9` percentage points.
- Window top-k match improves by about `10.7` percentage points.
- Region attribution remains high and stable.
- Top-k segment recall is lower than full train, but this is expected because the subset is contact-dense and each sequence has more GT segments competing for the same window caps.

## Segment Recall Interpretation

The subset top-k segment recall is:

```text
topk_gt_segment_recall = 68.6%
```

This is lower than full train, but not a blocker.

Likely reasons:

- The selected subset is contact-rich, so each sequence contains more GT contact segments.
- Actions such as `Carry on back`, `Sit on leg`, `Dance`, and `Hug` can include sustained or multi-part contact.
- The caps remain fixed:

```text
per_hand_max_windows = 2
per_seq_max_windows = 3
```

- More GT segments compete for a small number of local windows.

For refiner training, the objective is not to cover every GT segment. The objective is to produce enough high-quality local contact windows with clear supervision.

The subset meets that requirement:

```text
num_pred_windows = 6749
topk_window_match_ratio = 89.5%
window_contact_purity = 68.6%
false_positive_window_ratio = 15.6%
```

## Whether This Meets Expectations

Judgment:

```text
The 15-action contact-rich subset and rerun selector/window outputs meet expectations.
```

Reasons:

1. The subset is large enough:

```text
num_sequences = 2842
num_pred_windows = 6749
```

2. The subset is contact dense:

```text
num_gt_segments = 13190
gt_segments_per_sequence = 4.64
```

3. Sequence-level coverage is clean:

```text
gt_positive_zero_window_ratio = 0.0
gt_negative_nonzero_window_ratio = 0.0
```

4. Window-level quality is clearly better than full train:

```text
topk_window_match_ratio = 0.8947
window_contact_purity = 0.6857
false_positive_window_ratio = 0.1556
```

5. Region attribution remains reliable:

```text
topk_region_match_ratio = 0.9655
```

## Decision

Freeze the following for the first Stage2 refiner iteration:

```text
action subset = 15 selected action types
training bucket = GT+ / Pred+
selector config = hand-time top-k tau010 fixed
window artifact = refine_v2/outputs/train/contact_subset/selector_rerun/subset_selector_windows.npz
window metadata = refine_v2/outputs/train/contact_subset/selector_rerun/subset_window_metadata.json
```

Do not redesign selector/window before implementing the first refiner.

## Next Phase

Proceed to subset-focused refiner preparation:

1. Optional but recommended visual sanity check:

```text
vis_subset_windows.py
```

Suggested functionality:

- read `subset_selector_windows.npz`
- read `subset_selector_audit.json`
- read `subset_window_metadata.json`
- filter by action type, top-k match, false-positive flag, purity, region
- print/export timeline views and overlap summaries
- sample high-purity true positives, low-purity top-k matches, false positives, and per-action examples

2. Feature construction:

Minimal first feature pack should expose:

```text
actor_motion_window
coarse_motion_window
gt_motion_window
window mask / valid length
hand_side / hand_side_id
primary_target_region_id
topk_target_region_ids
topk_region_scores
dataset_row_index
sample_index
dataset_key
action_type
start_frame / end_frame
```

3. First refiner design:

Recommended first baseline:

```text
window-level residual refiner
input = actor + coarse window + hand/region condition
target = gt reactor window or residual = gt - coarse
output = reactor residual window
```

4. First loss design:

Possible first-scope losses:

```text
pose reconstruction loss on window
contact-weighted loss on GT contact frames
hand/region local loss
temporal smoothness loss
```

5. First eval design:

Minimum useful eval:

```text
window reconstruction error
contact frame error
region min-distance before/after
contact purity / contact distance before/after
subset action-type breakdown
```

Full sequence stitching can be deferred until the window-level refiner is working.

## Final Summary

The subset protocol is now good enough to become the Stage2 training domain.

The next productive work is:

```text
visual sanity check
-> feature dataset / window crop loader
-> minimal residual refiner
-> loss
-> window-level eval
```

Selector/window should remain fixed during the first refiner implementation.
