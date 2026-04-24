# Full-Sequence Eval Implementation

Date: 2026-04-24

## Goal

Implement the formal Stage2 system-level evaluation:

```text
Stage1 only (coarse)
vs
Stage1 + Stage2 (refined)
```

on stitched full sequences, with:

```text
STGCN / reconstruction metrics
contact metrics
```

computed from the same sampled subset sequences.

## Implemented Files

### 1. Stitching

```text
refine_v2/eval/full_sequence_stitch.py
```

Implements:

```text
- subset sequence selection
- balanced action-type sampling
- residual-space stitching from window predictions
- center-weighted merge over overlapping windows
- full-sequence refined reactor motion construction
```

Selected stitching rule:

```text
refined_full = coarse_full + merged_delta
```

where:

```text
merged_delta
= center-weighted average of overlapping window residuals
```

### 2. Full-sequence evaluation

```text
refine_v2/eval/full_sequence_eval.py
```

Implements:

```text
- contact-rich subset sampling
- stitched full-sequence pack construction
- STGCN eval via refine/eval/global_motion.py
- restored-space contact eval on GT / coarse / refined
- GT-relative surrogate penetration gap metrics
- per-sequence contact metrics
- action-type contact breakdown
```

### 3. CLI

```text
refine_v2/cli_eval_full_sequence.py
```

Outputs:

```text
full_sequence_eval.json
full_sequence_eval.md
full_sequence_eval_stgcn.csv
full_sequence_eval_contact.csv
optional full_sequence_eval_pack.npz
```

### 4. Commands

```text
refine_v2/commands/eval/17_eval_full_sequence_exp8_interaction_v1.sh
refine_v2/commands/visual/18_view_refiner_vis_pack_exp8_interaction_v1.sh
```

The visual command is a local convenience entry for the already-downloaded:

```text
refine_v2/save/train/refiner_v2_exp8_interaction_v1_10k/vis_pack_random20
```

## Protocol Implemented

The implementation follows the agreed protocol:

```text
1. final eval is full-sequence, not window-level
2. objects are GT / coarse / refined
3. STGCN is computed in canonical / Stage1-aligned processed space
4. contact is computed in restored pair space / restored shape
5. evaluation domain is the contact-rich subset
6. balanced sampled eval uses min(100, available) per action type
7. one unified pipeline produces both STGCN and contact reports
```

## Verification

Verified:

```text
python -m py_compile
python -m refine_v2.cli_eval_full_sequence --help
zsh -n on the new shell commands
```

One local smoke run was attempted, but the local workspace currently contains:

```text
exp8 eval outputs / vis pack
```

and does not contain:

```text
refine_v2/save/train/refiner_v2_exp8_interaction_v1_10k/model_best.pt
```

So the local smoke run stopped before model loading. This is a local artifact
availability issue, not a protocol/code-path issue.

## Next Step

Run the new full-sequence eval in the environment that has the checkpoint:

```text
bash refine_v2/commands/eval/17_eval_full_sequence_exp8_interaction_v1.sh
```

Then compare:

```text
Stage1 only vs Stage1 + Stage2
```

under:

```text
STGCN / reconstruction metrics
contact metrics
GT-relative penetration / overclose gaps
```
