# Table1 / Table2 Evaluation Extension Plan

Date: 2026-04-30

## Decision

The next evaluation-extension work should be split into two stages:

```text
Stage A:
  first complete the missing baseline results for table2

Stage B:
  then implement the batch Stage1 -> Stage2 bridge needed to complete table1
```

This ordering is preferred because table2 is much closer to the current
`refine_v2` evaluation pipeline, while table1 requires a new batch bridge from
sampled Stage1 outputs into Stage2.

## Table2: What already exists

Current `table2_stage2_stgcn_window_contact` is based on the frozen
`refine_v2` Stage2 protocol:

```text
train reaction_data
-> GT contact labels
-> fixed 15-action contact-rich subset
-> selector/window rerun on subset
-> Stage2 refiner training/eval
-> full-sequence evaluation
```

Important protocol details:

```text
1. evaluation domain = contact-rich subset
2. selector/window/subset are frozen
3. STGCN is computed in canonical / Stage1-aligned processed space
4. contact is computed in restored pair space / restored shape
5. full-sequence output is defined by residual-space center-weighted stitching
```

Current exp8 summary pages already correspond to this protocol:

```text
refine_v2/summary/evaluation_window.md
refine_v2/summary/evaluation_contact.md
refine_v2/summary/evaluation_stgcn.md
```

## Table2: Missing work

What is still missing for table2 is:

```text
add Stage1 baseline results on the same subset protocol
for:
  agrol
  mdm
  mdm-gru
  regennet
```

These should report:

```text
1. STGCN metrics
2. contact metrics
```

under the same subset/full-sequence protocol already used by exp8.

## Table2: Recommended implementation

For each baseline checkpoint:

```text
1. build reaction_data on dataset/interx/regen/train.h5
2. reuse the same subset_manifest.json
3. reuse the same full-sequence STGCN/contact evaluation protocol
4. treat the baseline output as coarse-only (no Stage2 refinement yet)
```

This is the lowest-risk extension because it stays inside the existing
`refine_v2` restored-space subset pipeline.

## Table1: What is different

`table1_stage1_stgcn` is not based on the subset pipeline above.

It is based on the Stage1 generative STGCN benchmark flow:

```text
checkpoint
-> sampling
-> STGCN evaluation
```

Therefore:

```text
the current table1 "hireact*" entry is still only Stage1 output
```

To obtain the true `hireact` result for table1, Stage1 sampled outputs must be
passed through Stage2 and then re-evaluated with STGCN.

## Table1: Required new bridge

The required new bridge is:

```text
Stage1 sampled results.npy/results_meta.npz
-> batch Stage2 input conversion
-> batch Stage2 exp8 refinement
-> refined STGCN evaluation
```

This cannot be recovered from the existing `eval_cmdm` yaml files alone,
because those files only store metrics, not the sampled motions needed for
Stage2 refinement.

## Existing code that can be reused

### For table2 extension

Reuse directly:

```text
refine.data.build_reaction_data
refine_v2/cli_eval_full_sequence.py
refine_v2/eval/full_sequence_eval.py
refine_v2/summary/evaluation_window.md
refine_v2/summary/evaluation_contact.md
refine_v2/summary/evaluation_stgcn.md
```

### For future table1 extension

Reuse partially:

```text
sample.cgenerate
eval/eval_cmdm.py
sample/infer_single_stage1_clip.py
refine/tools/build_reaction_data_from_results.py
```

These already prove that:

```text
Stage1 outputs can be packaged into Stage2-consumable inputs
```

but the table1 case still needs a proper batch implementation.

## Final Priority

Current agreed priority:

```text
1. first complete table2 baseline results on the subset protocol
2. then implement the batch bridge needed to complete table1 hireact
```

This keeps the next step narrow and reuses the already-stable `refine_v2`
evaluation pipeline before adding a new batch Stage1->Stage2 bridge.
