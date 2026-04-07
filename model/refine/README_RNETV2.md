# RNet V2 (Prior-Guided Local Contact Refinement)

This document summarizes the RNet v2-lite implementation in Stage2 refinement.
It is designed to keep Stage1 intact and improve local contact precision with
lightweight priors and evaluation.

## 1. Goal and Positioning

- Stage1: global motion distribution, realism, diversity.
- Stage2 (RNet v2): local residual refinement for contact precision and
  physical plausibility.

Core formula:

x_refined = x_coarse + M_active * delta_pred

Where delta_pred is a local residual (rot6d), and M_active is the active
window mask (time) plus joint mask (refine joints only).

## 2. Data Flow (Forward)

Inputs:
- actor_motion: [B, J, 6, T]
- coarse_motion: [B, J, 6, T]

Steps:
1) to_xyz -> actor_xyz, reactor_xyz: [B, J, 3, T]
2) ActiveWindowSelectorV2 -> active_mask [B, T], scores [B, T]
3) SurfaceFeatureBuilder -> geom_feat [B, T, Jr, Fg]
4) Extract coarse_local: [B, T, Jr, 6]
5) RNetV2Lite head -> delta [B, T, Jr, 6]
6) Scatter to delta_full [B, J, 6, T]
7) Apply masks: active_mask + joint_mask_full
8) refined = coarse_motion + delta_full

Outputs:
- refined
- aux: delta, active_mask, joint_mask, scores, geom_feat

## 3. Active Window Selector v2

Risk-aware score per frame:

score_t = alpha * dist_min - beta * approach - gamma * soft_contact

- dist_min: min actor-reactor distance among refine joints
- approach: relu(-(d_t - d_{t-1}))
- soft_contact: exp(-dist_min / sigma_contact)

Selection:
- keep top_k frames with smallest scores
- expand by window_size
- respect lengths and optional velocity gate

## 4. Geometry / Prior Features

Base geom feature (unchanged from v1): 9D joint-level features
- relative position (3)
- distance (1)
- relative velocity (3)
- soft contact (1)
- reactor speed (1)

Pairwise helpers (for loss/eval):
- build_pairwise_contact_stats
- build_distance_prior_targets

These compute distance and soft contact for selected joint pairs.

## 5. RNetV2Lite Head (Lightweight)

Modules:
- JointFeatureEmbed: [B, T, J, F] -> [B, T, J, H]
- TemporalConvBlock: per-joint depthwise temporal conv (residual)
- PartPoolingBlock: joint -> part pooling
- PartInteractionBlock: light attention + MLP on parts
- PartFusionBlock: part -> joint fusion
- OutputHead: [B, T, J, H] -> [B, T, J, 6]

## 6. Losses

Existing v1 losses:
- residual_loss
- residual_reg
- coordination_reg
- local_distance_loss (optional)

New v2 losses:
- distance_prior_loss: w * (d_refined - d_gt)^2, w = exp(-d_gt / tau)
- soft_contact_loss: (c_refined - c_gt)^2, c = exp(-d / sigma)
- smoothness_loss: ||delta_{t+1} - delta_t||^2

Masking:
- time mask uses lengths and active_mask
- smoothness uses adjacent frame intersection mask

Total loss:
L = lambda_res * L_res
  + lambda_reg * L_reg
  + lambda_coord * L_coord
  + lambda_contact * L_contact
  + lambda_dist * L_dist_prior
  + lambda_soft * L_soft_contact
  + lambda_smooth * L_smooth

## 7. CD (Contact Distance) Metric

GT-gated contact set:
Omega = {(t,p) | d_gt(t,p) < tau_contact}

CD = average over Omega of d_pred(t,p)

Outputs:
- cd_coarse, cd_refined, cd_improve
- cd_active_coarse, cd_active_refined, cd_active_improve

Aggregation:
- count-weighted average across batches
- safe if count == 0

## 8. Versioning and Compatibility

- v1 is preserved and still runnable.
- v2 path is enabled via rnet_version = v2
- Stage1 remains unchanged.

## 9. Notes and Limitations

- Current pairwise matching uses same-index joints (actor_i, reactor_i).
  Cross-joint or semantic pairs can be added later.
- CD tau_contact is fixed in eval for now.
- rnet_version is saved as config["version"] in checkpoints.

