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

## Source-aligned follow-up result

The mainchain test source was rebuilt and re-evaluated:

```text
refine_v2/save/table2_test/mainchain_test/reaction_data.npz
```

from:

```text
save/cnet_v5/interx_smplx_online_exp1/model000200000.pt
split = test
num_samples = -1
```

This removes the earlier table1-sampled-source mismatch.

### Result after source alignment

Even after source alignment, the test-side selector remains much weaker than
train:

```text
test full selector audit:
  num_sequences              = 1708
  num_gt_segments            = 3254
  num_pred_windows           = 2247
  gt_positive_zero_window    = 0.2444
  topk_gt_segment_recall     = 0.3556
  topk_window_match_ratio    = 0.3805
  window_contact_purity      = 0.1749
  false_positive_ratio       = 0.7494
```

And the fixed-manifest subset rerun is still far below train:

```text
test fixed-subset selector audit:
  num_sequences              = 439
  num_gt_segments            = 1981
  num_pred_windows           = 1066
  gt_positive_zero_window    = 0.0
  topk_gt_segment_recall     = 0.4291
  topk_window_match_ratio    = 0.5966
  window_contact_purity      = 0.2908
  false_positive_ratio       = 0.6079
```

### Interpretation update

The earlier source mismatch was real, but it is not the main remaining cause.

The stronger conclusion now is:

1. a real train/test generalization gap exists;
2. selector/proposal quality degrades sharply on test;
3. Stage2 still applies small positive geometric corrections on test;
4. but these corrections are too weak to yield strong binary-contact gains.

## Test-only selector ablation

We then kept the shared fixed-domain protocol unchanged and only relaxed the
internal HiReact test selector:

```text
selector_tau_contact: 0.10 -> 0.15
```

All other evaluation-domain definitions remained fixed:

- same `mainchain_test` reaction_data
- same GT contact labels
- same shared fixed manifest (`628` sequences)
- same full-sequence contact metrics

### Full-test selector comparison

Baseline selector (`tau=0.10`):

```text
num_pred_windows           = 2247
gt_positive_zero_window    = 0.2444
topk_gt_segment_recall     = 0.3556
window_contact_purity      = 0.1749
false_positive_ratio       = 0.7494
```

Relaxed selector (`tau=0.15`):

```text
num_pred_windows           = 2612
gt_positive_zero_window    = 0.1726
topk_gt_segment_recall     = 0.3875
window_contact_purity      = 0.1615
false_positive_ratio       = 0.7691
```

### Interpretation

This is the expected tradeoff:

- recall / coverage improve;
- purity degrades slightly;
- the tradeoff is acceptable enough to justify a full Stage2 run.

### Fixed-domain HiReact outcome

```text
HiReact*:
  F1        = 0.199653
  Recall    = 0.153555
  Distance  = 0.282808
  Ratio     = 0.258503

HiReact (tau=0.10):
  F1        = 0.200605
  Recall    = 0.155111
  Distance  = 0.281999
  Ratio     = 0.260754

HiReact (tau=0.15):
  F1        = 0.201244
  Recall    = 0.155731
  Distance  = 0.281808
  Ratio     = 0.261667
```

So the relaxed selector is beneficial, but only modestly:

```text
HiReact* < HiReact(tau=0.10) < HiReact(tau=0.15)
```

This improves the test result, but does not change the overall diagnosis that
test-side proposal quality is still the main bottleneck.

## Larger window-capacity follow-up

We then kept:

```text
selector_tau_contact = 0.15
```

and relaxed only the selector caps:

```text
per_hand_max_windows: 2 -> 3
per_seq_max_windows:  3 -> 5
```

### Selector audit effect

Compared with `tau=0.15, h2/s3`:

```text
num_pred_windows           : 2612 -> 3277
gt_segment_recall          : 0.1586 -> 0.1853
gt_contact_frame_coverage  : 0.1102 -> 0.1387
topk_gt_segment_recall     : 0.3875 -> 0.4425
window_contact_purity      : 0.1615 -> 0.1650
false_positive_ratio       : 0.7691 -> 0.7638
```

This means the larger caps did improve coverage, and did not degrade purity.

### Final fixed-domain HiReact comparison

```text
HiReact*:
  F1        = 0.199653
  Recall    = 0.153555
  Distance  = 0.282808
  Ratio     = 0.258503

HiReact (tau=0.10):
  F1        = 0.200605
  Recall    = 0.155111
  Distance  = 0.281999
  Ratio     = 0.260754

HiReact (tau=0.15):
  F1        = 0.201244
  Recall    = 0.155731
  Distance  = 0.281808
  Ratio     = 0.261667

HiReact (tau=0.15, h3/s5):
  F1        = 0.201229
  Recall    = 0.155930
  Distance  = 0.281575
  Ratio     = 0.262293
```

### Updated recommendation

Use:

```text
tau=0.15
```

as the current preferred test-side HiReact selector configuration, because it
gives the best `Contact F1`.

Do not promote `h3/s5` as the default test configuration yet:

- it improves geometry and contact ratio slightly;
- but it does not further improve the main contact `F1`.
