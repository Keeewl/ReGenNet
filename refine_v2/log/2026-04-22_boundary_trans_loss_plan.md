# Boundary Translation Loss And Next Iteration Plan

Date: 2026-04-22

## Context

The current refine_v2 refiner is a window-level model:

- each selector window is one independent training sample
- the model does not know other windows in the same sequence
- visualization stitches independent refined windows back into full sequences

The aitviewer check showed a new issue: when one sequence has multiple close or
overlapping windows, the refined reactor translation can jump at window
boundaries. The contact quality can improve, but full-sequence continuity can
look worse.

## Decision

Add a focused boundary translation anchor loss.

This iteration does not add `trans_vel` loss and does not change export
stitching. The goal is to address the issue at the model-output source first,
then re-run training, eval, and visualization.

## Implemented Loss

New train config fields:

```text
lambda_boundary_trans = 0.0 by default
boundary_trans_frames = 2 by default
```

Training command override:

```text
lambda_boundary_trans = 2.0
boundary_trans_frames = 2
```

Definition:

```text
L_boundary_trans =
  SmoothL1(pred_reactor_trans - coarse_reactor_trans)
  over first K and last K valid frames of each window
```

where:

```text
K = boundary_trans_frames
reactor translation = motion[55, :3, t]
```

Rationale:

- the boundary target is `coarse`, not GT, because the purpose is continuity
  with the full coarse sequence outside the window
- only translation is anchored because the visible artifact is global transl
  jump
- only first/last 1-2 frames are constrained so contact refinement inside the
  window remains free

Deferred:

- translation velocity loss
- sequence-level context model
- taper/weighted export stitching

## New Reference Metric

`eval_window` now also reports:

```text
coarse_boundary_trans_jump
pred_boundary_trans_jump
boundary_trans_jump_excess
```

This is a simple window-local reference metric. It measures translation step
size around the first and last frames of each window and compares refined
against coarse. It is not a full-sequence stitch metric, but it is cheap and
useful for checking whether boundary translation became less jumpy.

Primary interpretation:

```text
boundary_trans_jump_excess <= 0 is better
```

## New Training Command

```bash
bash refine_v2/commands/18_train_refiner_boundary_large.sh
```

This keeps the previous large exp2 architecture:

```text
hidden_dim = 512
num_heads = 8
num_layers = 8
mlp_ratio = 4.0
num_steps = 80000
```

and writes to:

```text
refine_v2/outputs/train/refiner_v2_exp3_boundary_large
```

## Evaluation After Training

Run the same eval pipeline as exp2, then export a new visualization pack:

```text
eval_window
eval_contact_refiner
export_refiner_vis_pack
aitviewer visual check
```

Key metrics to compare against exp2:

- `motion_improvement`
- `contact_motion_improvement`
- `gt_contact_dist_l1_improvement`
- `topk_gt_contact_dist_l1_improvement`
- `refined_contact_f1`
- `surrogate_penetration_depth_improvement`
- `boundary_trans_jump_excess`

The expected tradeoff:

- contact improvement may drop slightly
- window/full-sequence continuity should improve
- visible transl jumps should reduce

## Planning Under Time Pressure

Current efficient path:

1. Freeze selector/window/subset for now.
2. Train exp3 with boundary translation anchor.
3. Compare exp2 vs exp3 with `eval_window`, `eval_contact_refiner`, and aitviewer.
4. If exp3 preserves contact gains and reduces boundary jumps, treat this as the
   current stable baseline.
5. Only then upgrade feature/model/eval.

Recommended next upgrades, in priority order:

1. Better eval:
   - full-sequence stitch continuity metric
   - per-action contact/penetration summary
   - curated visualization packs for best/worst penetration and best/worst
     boundary jumps
2. Feature upgrade:
   - add normalized window phase / raw-segment position
   - add explicit boundary frame indicator
   - add current sequence/window overlap metadata if available
3. Model upgrade:
   - keep window-level model first
   - add stronger contact-aware conditioning before moving to sequence context
4. Loss upgrade:
   - consider weak `trans_vel` only if boundary anchor is insufficient
   - consider contact distance geometry loss only after offline contact eval
     proves it is stable enough

Avoid for now:

- redesigning selector/window
- switching to sequence-level refiner immediately
- adding many losses at once
- changing export stitching before measuring exp3

