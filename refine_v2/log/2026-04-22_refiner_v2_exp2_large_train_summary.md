# Refine V2 Exp2 Large Train Summary

Date: 2026-04-22

Run:

```text
save_dir = refine_v2/outputs/train/refiner_v2_exp2_large
model = RefineV2WindowRefiner
hidden_dim = 512
num_heads = 8
num_layers = 8
batch_size = 32
num_steps = 80000
warmup_steps = 1000
eval_interval = 2000
mixed_precision = true
lambda_region_dist = 0.0
```

Dataset:

```text
num_windows = 6749
train_windows = 6076
val_windows = 673
motion_shape = [56, 6, 30]
```

## Initial Val Baseline

At initialization, the output head is zero-initialized, so:

```text
pred_motion = coarse_motion
```

Initial heldout val metrics:

```text
coarse_motion_error = 0.016523013835188893
pred_motion_error   = 0.016523013835188893
motion_improvement  = 0.0

coarse_contact_motion_error = 0.016896587584782144
pred_contact_motion_error   = 0.016896587584782144
contact_motion_improvement  = 0.0

loss_total = 0.013519856919649362
```

This confirms the residual initialization is behaving as intended.

## Best Heldout Val Checkpoint

The best heldout val result occurred early, at step 4000:

```text
step = 4000

coarse_motion_error = 0.016523013835188893
pred_motion_error   = 0.015468684854027422
motion_improvement  = 0.0010543289811614717

coarse_contact_motion_error = 0.016896587584782144
pred_contact_motion_error   = 0.015771236569575003
contact_motion_improvement  = 0.0011253510152071404

loss_total = 0.011879286204324966
```

Relative improvement:

```text
motion improvement ~= 6.4%
contact-frame motion improvement ~= 6.7%
```

Interpretation:

- The refiner learns a real correction over the coarse baseline.
- The improvement is visible on heldout sequence-level val, not just train windows.
- This checkpoint is the currently meaningful model for downstream evaluation.

Use:

```text
refine_v2/outputs/train/refiner_v2_exp2_large/model_best.pt
```

Do not use `model_final.pt` as the main result for this run.

## Late Training / Overfitting

Training set metrics kept improving strongly through the end of the run.

Example near step 80000:

```text
coarse_motion_error = 0.01639184169471264
pred_motion_error   = 0.0054565174505114555
motion_improvement  = 0.010935324244201183

coarse_contact_motion_error = 0.017721261829137802
pred_contact_motion_error   = 0.005667995195835829
contact_motion_improvement  = 0.012053266167640686
```

This shows the model has enough capacity and the train loop/loss are effective.

However, heldout val degraded after the early best checkpoint.

Final val at step 80000:

```text
coarse_motion_error = 0.016523013835188893
pred_motion_error   = 0.0167810609633377
motion_improvement  = -0.0002580471281488087

coarse_contact_motion_error = 0.016896587584782144
pred_contact_motion_error   = 0.017033978527270666
contact_motion_improvement  = -0.00013739094248852438

loss_total = 0.013293227618131205
```

Interpretation:

```text
The 80k large run overfits.
```

The final/latest checkpoint is worse than the coarse baseline on heldout val.

## Full-Subset Eval

A separate `eval_window.json` was run over all 6749 subset windows:

```text
num_samples = 6749
num_batches = 211

coarse_motion_error = 0.016363784503580853
pred_motion_error   = 0.013494200193375494
motion_improvement  = 0.0028695843102053585

coarse_contact_motion_error = 0.016548299429070767
pred_contact_motion_error   = 0.013496039400354926
contact_motion_improvement  = 0.0030522600287158394

loss_total = 0.009376433264402871
```

Relative full-subset improvement:

```text
motion improvement ~= 17.5%
contact-frame motion improvement ~= 18.4%
```

Important caveat:

```text
This full-subset eval includes train windows, so it should not be interpreted
as heldout generalization.
```

It is useful for confirming that the trained refiner changes the window set in
the intended direction, but heldout val remains the primary model-selection
signal.

## Breakdown From Full-Subset Eval

Full-subset action-type improvements were positive across all listed actions.

Examples:

```text
Grab:              motion_improvement = 0.00201
Handshake:         motion_improvement = 0.00206
Kiss on cheek:     motion_improvement = 0.00240
Hug:               motion_improvement = 0.00290
Dance:             motion_improvement = 0.00353
Hand wrestling:    motion_improvement = 0.00358
Carry on back:     motion_improvement = 0.00369
Help up:           motion_improvement = 0.00403
```

Hand-side breakdown:

```text
right hand motion_improvement = 0.00283
left hand  motion_improvement = 0.00292
```

Primary-region breakdown:

```text
lower_body  motion_improvement = 0.00228
right_hand  motion_improvement = 0.00261
left_hand   motion_improvement = 0.00284
torso_head  motion_improvement = 0.00302
left_arm    motion_improvement = 0.00308
right_arm   motion_improvement = 0.00311
```

This suggests the model is not only improving one narrow region/action group,
but this conclusion is still train-contaminated because the full subset includes
training windows.

## Conclusion

The first refiner direction is validated:

- overfit test passed
- train loss decreases strongly
- model can improve over coarse motion
- best heldout val checkpoint improves both full-window motion and contact-frame motion

But the large 80k run is too long for the current regularization/data scale:

- best val occurs at step 4000
- later training overfits
- final checkpoint is worse than coarse on heldout val

Current best model:

```text
refine_v2/outputs/train/refiner_v2_exp2_large/model_best.pt
```

Do not use:

```text
model_final.pt
model_latest.pt
```

as the primary result for this run.

## Next Training Recommendation

Run a shorter, better-regularized experiment:

```text
hidden_dim = 384 or 512
num_layers = 6
num_heads = 8
num_steps = 8000 to 12000
eval_interval = 500 or 1000
dropout = 0.15 to 0.20
weight_decay = 5e-4 to 1e-3
lambda_smooth = 0.10
lambda_region_dist = 0.0
```

Primary model-selection metric:

```text
heldout val pred_motion_error
```

Secondary metric:

```text
heldout val pred_contact_motion_error
```

Training should stop early if heldout val improvement degrades for several eval
intervals.

The next useful implementation improvement is early stopping / patience in the
trainer, so long overnight runs do not spend most of their time overfitting.
