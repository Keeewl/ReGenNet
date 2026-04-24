# Window Evaluation

> Note: the values shown on this page are from `exp8` and currently represent the best Stage2 result in this repo.

## Scope

This page summarizes the frozen Stage2 window-selection quality on the
contact-rich subset.

The selected main metrics are:

```text
num_sequences
num_gt_segments
num_pred_windows

gt_positive_zero_window_ratio
topk_window_match_ratio
topk_region_match_ratio
window_contact_purity
```

These metrics describe:

```text
1. how much GT contact content exists in the subset
2. how many windows the selector proposes
3. whether GT-positive sequences are missed
4. whether predicted windows really correspond to GT contact
5. whether predicted top-k regions are semantically aligned
6. how pure the selected windows are internally
```

## Current Frozen Selector/Window Result

```text
num_sequences = 2842
num_gt_segments = 13190
num_pred_windows = 6749

gt_positive_zero_window_ratio = 0.0
topk_window_match_ratio = 0.8947
topk_region_match_ratio = 0.9655
window_contact_purity = 0.6857
```

## Metric Definitions

### num_sequences

```text
the number of subset sequences entering the final selector/window evaluation
```

Current value:

```text
2842
```

### num_gt_segments

```text
the number of GT contact segments in the evaluated subset
```

More precisely:

```text
GT contact segments are built at the sample × hand × region level
from binary GT contact masks
```

Current value:

```text
13190
```

### num_pred_windows

```text
the number of final fixed windows output by the selector
```

This is after:

```text
raw segment formation
window generation
per-hand / per-sequence caps
```

Current value:

```text
6749
```

### gt_positive_zero_window_ratio

Definition:

```text
among GT-positive sequences,
the ratio of sequences that end up with zero predicted windows
```

Interpretation:

```text
this is the most important sequence-level coverage indicator
```

Current value:

```text
0.0
```

Meaning:

```text
the frozen selector does not completely miss any GT-positive sequence
in the contact-rich subset
```

### topk_window_match_ratio

Definition:

```text
the fraction of predicted windows that match at least one GT contact segment
under same-sample + same-hand + time-overlap,
with GT region falling inside the predicted top-k regions
```

Interpretation:

```text
this is a practical window-level precision metric
```

Current value:

```text
0.8947
```

Meaning:

```text
about 89.5% of selected windows correspond to real GT contact content
```

### topk_region_match_ratio

Definition:

```text
among predicted windows that already have same-hand / time overlap with GT,
the fraction whose predicted top-k regions contain the GT region
```

Interpretation:

```text
this measures whether the selector's region attribution is semantically correct
```

Current value:

```text
0.9655
```

Meaning:

```text
once the selector finds the correct hand/time event,
the top-k region attribution is almost always correct
```

### window_contact_purity

Definition:

```text
the fraction of frames inside predicted windows that are true GT contact frames
```

Interpretation:

```text
this measures how internally pure each selected window is
```

Current value:

```text
0.6857
```

Meaning:

```text
on average, about 68.6% of the frames inside selected windows are true GT contact
```

## Overall Reading

The current frozen selector/window result can be summarized as:

```text
1. it does not miss GT-positive sequences
2. most predicted windows correspond to real GT contact
3. top-k region attribution is very accurate
4. the selected windows are reasonably contact-pure internally
```

So the current selector/window module is strong enough to serve as the frozen
Stage2 proposal stage.
