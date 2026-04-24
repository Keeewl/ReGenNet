# Contact Evaluation

> Note: the values shown on this page are from `exp8` and currently represent the best Stage2 result in this repo.

## Scope

Stage2 contact evaluation should be interpreted as:

```text
GT / coarse / refined
```

where:

```text
GT       = restored-space binary / geometric contact reference
coarse   = Stage1 output
refined  = Stage1 + Stage2 output
```

The current main system-level contact report is based on:

```text
full-sequence evaluation
contact-rich subset
residual-space center-weighted window stitching
restored pair space / restored shape contact computation
```

## Binary Contact Metrics

Binary contact uses:

```text
region-to-region minimum distance < tau_contact
```

with:

```text
tau_contact = 0.05
```

The main binary metrics are:

```text
F1
precision
recall
```

Interpretation:

```text
precision = among predicted contact frames, how many are true GT contact frames
recall    = among GT contact frames, how many are recovered
F1        = harmonic mean of precision and recall
```

For GT itself, these metrics are not reported as a meaningful row because:

```text
GT compared to GT would trivially be 1.0 / 1.0 / 1.0
```

So the effective table is:

| variant | contact_f1 | contact_precision | contact_recall |
| --- | --- | --- | --- |
| GT | reference | reference | reference |
| coarse | 0.781628 | 0.841105 | 0.730007 |
| refined | 0.794553 | 0.845477 | 0.749415 |

Current reading:

```text
1. refined contact F1 is clearly higher than coarse
2. refined precision is slightly higher than coarse
3. refined recall improves more noticeably than precision
4. Stage2 mainly helps recover more true GT contact frames without introducing a large precision drop
```

## Metric Selection

Recommended core contact metrics for main reporting:

```text
1. contact_f1
2. contact_precision
3. contact_recall
4. Contact ratio
5. Average contact duration
6. Contact distance
```

## Contact Ratio

`Contact ratio` is the fraction of valid frames that are contact frames.

Suggested display:

| variant | Contact ratio |
| --- | --- |
| GT | 0.489563 |
| coarse | 0.442139 |
| refined | 0.447734 |

Interpretation:

```text
1. this is an absolute value, not an improvement value
2. it measures how much of the sequence is in contact
3. refined is closer to GT than coarse on overall contact amount
```

## Average Contact Duration

`Average contact duration` is the mean duration of contiguous contact segments.

Suggested display:

| variant | Average contact duration |
| --- | --- |
| GT | 38.220861 |
| coarse | 33.272267 |
| refined | 33.523666 |

Interpretation:

```text
1. this is an absolute value, not an improvement value
2. it measures how long each contact segment lasts on average
3. refined is slightly closer to GT than coarse on contact duration
4. there is still a visible gap to GT contact persistence
```

## Contact Distance

For direct geometric reporting, use:

```text
Contact distance
```

This should be defined as:

```text
mean hand-target minimum distance
computed only on GT-contact regions / frames
```

In implementation terms, the displayed table corresponds to:

```text
GT      -> gt_contact_gt_min_dist
coarse  -> gt_contact_coarse_min_dist
refined -> gt_contact_refined_min_dist
```

Suggested display:

| variant | Contact distance |
| --- | --- |
| GT | 0.016226 |
| coarse | 0.034261 |
| refined | 0.032419 |

Interpretation:

```text
1. lower is better
2. this is not a global average over all frames
3. it is computed only on GT-contact frames / regions
4. refined closer to GT than coarse indicates better contact refinement
```

Current exp8 full-sequence reading:

```text
GT      = 0.016226
coarse  = 0.034261
refined = 0.032419
```

So:

```text
refined is closer to GT than coarse
but there is still a clear remaining gap to GT
```
