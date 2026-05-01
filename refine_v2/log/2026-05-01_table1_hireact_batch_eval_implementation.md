# Table1 HiReact Batch Eval Implementation

Date: 2026-05-01

## Goal

Complete the remaining implementation needed to turn the one-seed `table1` HiReact dry-run into the final 20-seed table row.

## Context

The one-seed bridge was already validated:

```text
Stage1 sampled outputs
-> Stage2 exp8 refine
-> train-conditioned / test-conditioned STGCN
```

The remaining work was:

```text
1. parameterize the run by seed
2. batch-run seeds 0..19
3. aggregate mean +/- interval using the same table1 code path convention
```

## Added Pieces

### 1. Seed-parameterized one-run command

Already present and now treated as the unit run:

```text
refine_v2/commands/eval/25_run_table1_hireact_seed.sh
```

This command:

```text
seed -> train reaction_data
     -> test reaction_data
     -> table1 HiReact dry-run eval
```

### 2. Batch seed runner

Added:

```text
refine_v2/commands/eval/26_run_table1_hireact_seeds_0_19.sh
```

This loops:

```text
SEED = 0..19
```

and invokes the per-seed command for each run.

### 3. Final aggregation tool

Added:

```text
refine_v2/tools/aggregate_table1_hireact.py
refine_v2/cli_aggregate_table1_hireact.py
refine_v2/commands/eval/27_aggregate_table1_hireact.sh
```

This aggregator:

- reads all one-seed summary files
- extracts `train_conditioned` and `test_conditioned` refined STGCN metrics
- aggregates:
  - `fid`
  - `accuracy`
  - `diversity`
  - `multimodality`
- preserves the current table1 interval convention from `eval/easy_table.py`:

```text
interval = 1.96 * var(values)
```

Outputs:

```text
table1_hireact_aggregate.json
table1_hireact_aggregate.csv
table1_hireact_aggregate.md
```

## Path Fix

The aggregator default summary glob was aligned with the actual one-seed output structure:

```text
refine_v2/save/table1/hireact_seed*/hireact_dryrun/table1_hireact_dryrun_summary.json
```

## Verification

Passed:

```text
python3 -m py_compile refine_v2/tools/aggregate_table1_hireact.py refine_v2/cli_aggregate_table1_hireact.py
zsh -n refine_v2/commands/eval/25_run_table1_hireact_seed.sh
zsh -n refine_v2/commands/eval/26_run_table1_hireact_seeds_0_19.sh
zsh -n refine_v2/commands/eval/27_aggregate_table1_hireact.sh
```

## Current Status

The implementation side is now complete for:

```text
table1 HiReact:
1-seed validated
20-seed batch command available
final aggregation command available
```

The remaining work is only to run:

```text
26_run_table1_hireact_seeds_0_19.sh
27_aggregate_table1_hireact.sh
```
