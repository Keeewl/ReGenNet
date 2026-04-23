# exp6 Phase-Smallroot Implementation

Date: 2026-04-23

Implemented experiment:

```text
refiner_v2_exp6_phase_smallroot_10k
```

## Implemented Changes

### Phase-Aware Preserve Loss

Updated:

```text
refine_v2/model/losses_v2.py
refine_v2/model/joint_groups.py
refine_v2/train/trainer.py
refine_v2/cli_train_refiner.py
refine_v2/cli_eval_refiner.py
```

New config parameters:

```text
lambda_phase_preserve
phase_preserve_power
phase_preserve_transl_weight
phase_preserve_root_weight
phase_preserve_lower_body_weight
phase_preserve_torso_weight
phase_preserve_arm_weight
phase_preserve_hand_weight
```

Loss definition:

```text
phase_weight[t] = (abs(t - center) / center) ** phase_preserve_power

L_phase_preserve =
  mean phase_weight[t] * group_weight[j] * SmoothL1(pred[j,t], coarse[j,t])
```

Default behavior remains unchanged because:

```text
lambda_phase_preserve = 0.0
```

exp6 enables it with:

```text
lambda_phase_preserve = 0.5
phase_preserve_power = 2.0
phase_preserve_transl_weight = 2.0
phase_preserve_root_weight = 1.0
phase_preserve_lower_body_weight = 0.5
phase_preserve_torso_weight = 0.3
phase_preserve_arm_weight = 0.1
phase_preserve_hand_weight = 0.05
```

Purpose:

```text
allow more correction near the window center
preserve coarse motion near window boundaries
mainly constrain transl/root, not hand/arm
```

### Metrics

Training/eval now reports:

```text
loss_phase_preserve
```

### Commands

Added:

```text
refine_v2/commands/train/03_train_refiner_phase_smallroot.sh
refine_v2/commands/eval/05_eval_refiner_phase_smallroot.sh
refine_v2/commands/eval/06_eval_contact_refiner_phase_smallroot.sh
refine_v2/commands/visual/06_export_refiner_vis_pack_phase_smallroot.sh
refine_v2/commands/visual/07_diagnose_refiner_vis_pack_phase_smallroot.sh
```

Output path:

```text
refine_v2/save/train/refiner_v2_exp6_phase_smallroot_10k
```

## exp6 Command Defaults

Key changes vs exp5:

```text
hand_delta_scale = 1.2
root_delta_scale = 0.25
transl_delta_scale = 0.30

selected_hand_motion_weight = 3.5
selected_hand_contact_weight = 5.0

lambda_boundary_trans = 1.0
lambda_phase_preserve = 0.5
```

Kept from exp5:

```text
use_geometry_features
use_group_gated_residual
use_group_weighted_loss
use_hand_arm_contact_loss
hidden_dim = 512
num_layers = 8
num_heads = 8
num_steps = 10000
```

## Validation

Completed:

```text
python3 -m py_compile refine_v2/model/joint_groups.py refine_v2/model/losses_v2.py refine_v2/train/trainer.py refine_v2/cli_train_refiner.py refine_v2/cli_eval_refiner.py
python3 -m refine_v2.cli_train_refiner --help
python3 -m refine_v2.cli_eval_refiner --help
```

Smoke tests:

```text
geometry-enabled model forward + exp6 model config + exp6 loss config: passed
direct nonzero phase-preserve loss test: passed
```

Note:

```text
loss_phase_preserve is zero at model initialization if pred == coarse,
which is expected because the output head is zero-initialized.
```

## Next Run

Run:

```text
bash refine_v2/commands/train/03_train_refiner_phase_smallroot.sh
```

Then:

```text
bash refine_v2/commands/eval/05_eval_refiner_phase_smallroot.sh
bash refine_v2/commands/eval/06_eval_contact_refiner_phase_smallroot.sh
bash refine_v2/commands/visual/06_export_refiner_vis_pack_phase_smallroot.sh
bash refine_v2/commands/visual/07_diagnose_refiner_vis_pack_phase_smallroot.sh
```

Compare against exp5:

```text
refined_contact_f1
topk_refined_contact_f1
gt_contact_contact_dist_improvement
boundary_trans_jump_excess
delta_norm_transl
refined_topk_gap_to_gt
diagnosis_ratio_already_good
```

## Run Result

The exp6 run completed and was evaluated.

Detailed result log:

```text
refine_v2/log/2026-04-23_exp6_phase_smallroot_eval_summary.md
```

Main result:

```text
exp6 does not beat exp5.
```

Key exp6 contact metrics:

```text
gt_contact_contact_dist_improvement = 0.0022491207
refined_contact_f1 = 0.8172011076
topk_refined_contact_f1 = 0.8250681969
surrogate_penetration_depth_improvement = -0.0000531603
```

exp5 reference:

```text
gt_contact_contact_dist_improvement = 0.0028254371
refined_contact_f1 = 0.8221591739
topk_refined_contact_f1 = 0.8297871497
surrogate_penetration_depth_improvement = -0.0000661652
```

Conclusion:

```text
phase preserve is useful as an available loss option,
but the exp6 setting is too conservative and weakens hand/arm contact
correction.
```

Decision:

```text
keep exp5 as the practical baseline
do not adopt exp6 phase-smallroot as the next baseline
```
