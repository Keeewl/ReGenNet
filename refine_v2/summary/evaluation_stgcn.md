# STGCN Evaluation

> Note: the values shown on this page are from `exp8` and currently represent the best Stage2 result in this repo.

## Scope

Stage2 STGCN evaluation should be interpreted as:

```text
GT / coarse / refined
```

where:

```text
GT       = ground-truth full sequence
coarse   = Stage1 output
refined  = Stage1 + Stage2 output
```

Important protocol:

```text
STGCN metrics are computed in canonical / Stage1-aligned processed space
after inverse restore
```

This means:

```text
Stage2 refinement happens in restored pair space
but STGCN is evaluated after mapping the result back to the Stage1 motion space
```

## Current exp8 Full-Sequence STGCN Result

| variant | FID↓ | Acc.↑ | Div.→ | Multimod.→ |
| --- | --- | --- | --- | --- |
| GT | ~0 | 0.985878 | 21.559633 | 4.758754 |
| coarse | 0.474216 | 0.977808 | 21.202896 | 4.836596 |
| refined | 0.295525 | 0.978480 | 21.273443 | 4.820340 |

## Metric Definitions

### Accuracy

```text
STGCN action-recognition accuracy on the evaluation sequences
```

Interpretation:

```text
higher is better
```

Current reading:

```text
GT       = 0.985878
coarse   = 0.977808
refined  = 0.978480
```

So:

```text
refined is slightly better than coarse on STGCN accuracy
```

### Diversity

```text
within-set STGCN feature diversity
```

Interpretation:

```text
it measures how spread out the motions are in STGCN feature space
```

Current reading:

```text
GT       = 21.559633
coarse   = 21.202896
refined  = 21.273443
```

So:

```text
refined is slightly closer to GT than coarse on feature diversity
```

### Multimodality

```text
same-label STGCN feature distance
```

Interpretation:

```text
it measures intra-class variety in STGCN feature space
```

Current reading:

```text
GT       = 4.758754
coarse   = 4.836596
refined  = 4.820340
```

So:

```text
refined is also slightly closer to GT than coarse
```

### FID

```text
GT-referenced Fréchet distance in STGCN feature space
```

Interpretation:

```text
lower is better
GT is approximately zero by definition
```

Current reading:

```text
GT       = ~0
coarse   = 0.474216
refined  = 0.295525
```

So:

```text
refined is significantly closer to GT than coarse in STGCN feature space
```

## Overall Reading

The current exp8 full-sequence STGCN result can be summarized as:

```text
1. Stage2 does not damage global motion distribution
2. refined is slightly better than coarse on STGCN accuracy
3. refined is slightly better than coarse on diversity / multimodality alignment
4. refined is clearly better than coarse on STGCN FID
```

Under the Stage2 objective:

```text
STGCN is an auxiliary system-level metric
contact remains the primary target
```

So the practical conclusion is:

```text
Stage2 exp8 improves contact while also keeping, and slightly improving,
global STGCN motion quality
```
