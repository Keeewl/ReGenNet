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
