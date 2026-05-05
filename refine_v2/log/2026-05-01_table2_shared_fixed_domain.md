# 2026-05-01 Table2 Shared Fixed-Domain Redesign

## Decision

We are changing `table2` from a per-method `GT+ / Pred+` subset protocol to a
shared fixed-domain protocol.

### Old issue

The previous `table2` extension used each method's own selector windows to
define `Pred+`, which meant:

- different baselines were evaluated on different sequence subsets;
- this was acceptable for internal Stage2 task analysis;
- it is not a clean shared benchmark table across methods.

### New rule

For each split (`train`, `test`), `table2` now uses:

1. the same fixed 15 Inter-X contact-rich action labels;
2. the same fixed sequence set for every method in that split;
3. full-sequence restored-space contact metrics only.

Selector windows remain necessary only for `HiReact` inference, not for
defining which sequences are evaluated.

## Implementation

Added:

- `refine_v2/tools/build_fixed_eval_manifest.py`
- `refine_v2/cli_build_fixed_eval_manifest.py`
- `refine_v2/commands/table2_fixed/01_build_train_fixed_manifest.sh`
- `refine_v2/commands/table2_fixed/02_build_test_fixed_manifest.sh`

The manifest builder:

- reads action metadata from `reaction_data`;
- filters by the explicit 15 action labels;
- writes a shared `bucket_label=FIXED` manifest;
- does not depend on selector windows, selector audit, or `Pred+`.

## Consequence for methods

### Stage1-only baselines

- `AGRoL`
- `MDM`
- `MDM-GRU`
- `ReGenNet`
- `HiReact*`

Pipeline becomes:

```text
reaction_data -> GT contact labels -> fixed manifest -> coarse_only eval
```

### HiReact

Pipeline becomes:

```text
reaction_data -> fixed manifest -> internal selector/windows -> geometry cache -> Stage2 refine -> full-sequence eval
```

The selector still exists, but it is no longer allowed to choose the evaluation
domain.

## Current fixed-domain test results

The shared fixed-domain manifests were built successfully:

```text
train fixed manifest: 3400 sequences
test fixed manifest:   628 sequences
```

Current completed test rows:

### AGRoL

```text
Contact F1      = 0.111718
Recall          = 0.084207
Contact Distance= 0.350403
Contact Ratio   = 0.262590
```

### MDM

```text
Contact F1      = 0.135332
Recall          = 0.099289
Contact Distance= 0.352129
Contact Ratio   = 0.237495
```

### MDM-GRU

```text
Contact F1      = 0.127428
Recall          = 0.092421
Contact Distance= 0.392654
Contact Ratio   = 0.227272
```

### ReGenNet

```text
Contact F1      = 0.169020
Recall          = 0.126585
Contact Distance= 0.307416
Contact Ratio   = 0.246582
```

### HiReact*

```text
Contact F1      = 0.199653
Recall          = 0.153555
Contact Distance= 0.282808
Contact Ratio   = 0.258503
```

### HiReact

```text
Contact F1      = 0.200605
Recall          = 0.155111
Contact Distance= 0.281999
Contact Ratio   = 0.260754
```

### HiReact (test-only selector tau=0.15)

```text
Contact F1      = 0.201244
Recall          = 0.155731
Contact Distance= 0.281808
Contact Ratio   = 0.261667
```

This run keeps the shared fixed-domain evaluation set unchanged and only
relaxes the internal HiReact selector threshold on test.

### HiReact (test-only selector tau=0.15, h3/s5)

```text
Contact F1      = 0.201229
Recall          = 0.155930
Contact Distance= 0.281575
Contact Ratio   = 0.262293
```

This larger-cap selector variant improves geometric/contact coverage slightly,
but does not beat `tau=0.15` on the main `Contact F1` metric.

## Current conclusion

The shared fixed-domain result preserves the same qualitative picture:

- all baseline test rows are now complete under the shared fixed-domain protocol;
- test is substantially harder than train for every method;
- `HiReact*` is stronger than every Stage1 baseline on test;
- `HiReact*` is stronger than `ReGenNet` on test;
- `HiReact` is slightly stronger than `HiReact*`;
- a test-only selector relaxation to `tau=0.15` improves HiReact further, but
  only modestly;
- further increasing window caps (`h3/s5`) improves coverage and geometry but
  does not further improve the main contact `F1`;
- Stage2 still provides only a small gain on test;
- therefore the dominant bottleneck is still weak Stage1/selector behavior on
  test rather than a trivial Stage2 loading bug.
