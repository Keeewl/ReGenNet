# Table2 Test Contact Gap Diagnosis

Date: 2026-05-01

## Immediate finding

The current `table2` test-side contact result is not directly comparable to the
existing train-side table2 result.

## Why

### Train table2 source

Train table2 is built from:

```text
refine/dataset/train/reaction_data.npz
```

which comes from:

```text
save/cnet_v5/interx_smplx_online_exp1/model000200000.pt
```

with the original Stage2 main-chain build.

### Current test table2 source

The current test-side contact chain was built from:

```text
refine_v2/save/table1/dryrun_01/cnetv5_seed0_test/reaction_data.npz
```

which comes from the table1 benchmark bridge:

```text
save/cnet_v5_256/interx_smplx_online_exp1/model000209455.pt
num_samples = 1000
seed = 0
```

This means current train/test table2 contact numbers are mixing:

```text
different Stage1 checkpoints
different sample domains
different source protocols
```

## Additional evidence

Table1 itself already shows a large train/test gap for Stage1/STGCN, so a real
generalization gap clearly exists. But the current table2 test contact result
adds a second confound:

```text
table1 sampled source -> table2 contact protocol
```

This must be removed before deciding whether the failure is mainly:

1. Stage1 domain gap,
2. selector/window degradation,
3. Stage2 weak refinement on test,
4. or a restored-space / mapping bug.

## Current strongest signal

The strongest current signal is selector degradation on test:

```text
train subset selector:
  topk_gt_segment_recall  ~ 0.686
  topk_window_match_ratio ~ 0.895
  window_contact_purity   ~ 0.686
  false_positive_ratio    ~ 0.156

test subset selector:
  topk_gt_segment_recall  ~ 0.442
  topk_window_match_ratio ~ 0.580
  window_contact_purity   ~ 0.332
  false_positive_ratio    ~ 0.582
```

So even before touching refiner details, the test-side windows are already much
weaker.

## First ranked fix

Rebuild test reaction_data from the same Stage1 main chain as train:

```text
save/cnet_v5/interx_smplx_online_exp1/model000200000.pt
split = test
num_samples = -1
```

This removes the table1-vs-table2 source mismatch.

## Added command

```text
refine_v2/commands/eval/28_build_test_reaction_data_mainchain.sh
```

This should be the next command to run before further test-contact conclusions.
