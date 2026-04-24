# Full-Sequence Eval Protocol for Stage2

Date: 2026-04-24

## Goal

Define the formal system-level evaluation protocol for Stage2 refine_v2.

The core principle is:

```text
final Stage2 evaluation should be full-sequence evaluation,
not window-level evaluation
```

Window-level metrics are still useful for local debugging, but the final
system-level comparison must be based on stitched full-sequence outputs.

## 1. Evaluation Objects

The full-sequence evaluation should always compare:

```text
GT
coarse (Stage1 output)
refined (Stage1 + Stage2 output)
```

This gives the correct comparisons:

```text
coarse vs GT
refined vs GT
refined relative to coarse
```

## 2. Stitching Rule

Stage2 predicts window-level residual refinements, so full-sequence output
must be defined by stitching the residuals back to the full reactor sequence.

The selected rule is:

```text
merge in residual space
use center-weighted merge over overlapping windows
frames not covered by any window keep the coarse motion unchanged
```

More concretely:

```text
refined_full = coarse_full + merged_delta
```

where:

```text
merged_delta[t] =
  weighted average of all window residual predictions covering frame t
```

The weighting should prioritize window centers over window edges.

This is preferred over simple averaging because:

```text
window centers are typically more reliable
window edges are more affected by truncation / boundary effects
```

## 3. Metric Spaces

The two evaluation families should be computed in different spaces:

### A. STGCN / reconstruction metrics

Use:

```text
canonical motion space
```

Reason:

```text
STGCN and reconstruction metrics are motion-distribution / reconstruction
metrics, not contact-geometry metrics
```

### B. Contact metrics

Use:

```text
restored pair space with restored shape
```

Reason:

```text
contact metrics must measure actual actor-reactor geometry and mesh-region
contact in restored pair space
```

## 4. Evaluation Domain

The formal Stage2 report should use:

```text
contact-rich subset
```

Reason:

```text
Stage2 is trained and designed as a contact-refine module on the contact-rich
subset, not as a full-train universal motion reconstructor
```

Therefore:

```text
reporting on the subset is aligned with the Stage2 task definition
```

Important clarification:

```text
subset-level reporting is the main Stage2 claim
it should not be confused with a whole-train universal evaluation claim
```

## 5. Sampling Protocol

The recommended primary reporting protocol is:

```text
balanced sampled eval by action type
```

Specifically:

```text
for each selected action type:
  use min(100, available_sequences_of_that_type)
```

This is preferred because:

```text
1. it avoids large action types dominating the summary
2. it gives each contact-rich action type a fair contribution
3. it keeps computation manageable
4. it produces a more interpretable report
```

## 6. Reporting Structure

The system-level evaluation should be implemented in one unified pass, but the
output should be split into two report tables:

### A. STGCN / reconstruction table

Example contents:

```text
GT / coarse / refined
STGCN
motion reconstruction errors
optional smoothness / velocity metrics
```

### B. Contact table

Example contents:

```text
GT / coarse / refined
contact F1
top-k contact F1
GT-contact distance improvement
GT-relative penetration / overclose gap
```

This should be:

```text
one evaluation pipeline
two metric families
two separate report tables
```

This is preferred over two independent scripts because:

```text
1. the evaluated sample set stays exactly aligned
2. the input pack stays consistent
3. Stage1-only vs Stage1+Stage2 comparisons are easier to audit
```

## 7. Stage2 Objective Interpretation

This full-sequence protocol should follow the corrected Stage2 objective:

```text
Stage2 is a contact-refine module, not a motion-reconstruction module
```

Therefore:

```text
1. weaker STGCN / reconstruction is acceptable
2. contact metrics are the primary success criteria
3. penetration must be interpreted relative to GT, not as an absolute-minimize metric
```

The correct system-level question is:

```text
does Stage1 + Stage2 produce meaningfully better contact than Stage1 only,
even if reconstruction metrics degrade somewhat
```

## 8. Final Protocol Summary

```text
full-sequence evaluation
GT / coarse / refined comparison
residual-space center-weighted stitching
STGCN in canonical space
contact in restored pair space
subset-only reporting
balanced action-type sampling: min(100, available)
one unified evaluation pass
two report tables: STGCN and contact
```
