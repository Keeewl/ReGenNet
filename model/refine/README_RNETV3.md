# RNet V3 (Physics-First Contact Refinement)

This document summarizes the RNet v3 implementation for Stage2 refinement.
It keeps Stage1 intact and stays lightweight at inference time.

## 1. Goal and Positioning

- Stage1: global motion distribution, realism, diversity.
- Stage2 (RNet v3): local residual refinement prioritizing contact distance (CD) and physics contact quality.

Core formula:

```
x_refined = x_coarse + M_active * (gate * delta_bounded)
```

Where delta is bounded and gated, and M_active is the active window mask (time) plus joint mask (refine joints only).

## 2. Data Flow (Forward)

Inputs:
- actor_motion: [B, J, 6, T]
- coarse_motion: [B, J, 6, T]

Steps:
1) to_xyz -> actor_xyz, reactor_xyz: [B, J, 3, T]
2) ActiveWindowSelectorV2 -> coarse-risk mask (inference)
3) (train only) Oracle-enhanced mask: M_gt_contact ∪ M_contact_error ∪ M_coarse_risk
4) SurfaceFeatureBuilder -> geom_feat [B, T, Jr, F] (9D base + optional contact aug)
5) RNetV3Lite head -> delta_raw + gate_logits
6) delta_bounded = delta_max * tanh(delta_raw / delta_max)
7) gate = sigmoid(gate_logits), delta = gate * delta_bounded
8) Scatter + apply masks -> refined

## 3. Active Window (Train vs Inference)

- Inference: coarse-risk selector only (no GT).
- Training: oracle-enhanced mask
  - M_gt_contact: GT contact/near-contact frames
  - M_contact_error: coarse vs GT contact mismatch frames
  - M_coarse_risk: selector proposal

Window size semantics: the selector uses a `window_size` and expands by `radius = window_size // 2`.
This means a window_size of 7 covers roughly 7 frames (center +/- 3).

Overlap diagnostics (logged during training):
- overlap_iou
- gt_contact_recall_by_coarse_risk
- coarse_risk_precision_wrt_gt

## 4. Semantic-Nearest Pairwise Contact

- v3 uses semantic candidate pairs instead of same-index pairing.
- For each candidate part-pair, the top-k nearest joint pairs are selected per frame.
- Default focus: upper-body and hands (no dense mesh / SDF).

Default candidates (simplified):
- actor hand ↔ reactor hand / arm / torso_head
- actor arm ↔ reactor hand / arm / torso_head

## 5. Contact-Oriented Feature Augmentation

Base 9D geometry features are preserved:
- relative position (3)
- distance (1)
- relative velocity (3)
- soft contact (1)
- reactor speed (1)

Optional lightweight augmentation (v3 default on):
- nearest distance stats (top1, topk mean, margin)
- closing speed (temporal derivative of nearest distance)
- part-level contact summary (mean soft contact)

## 6. Physics-First Losses

Loss groups:
- Physics terms (primary):
  - distance_prior_loss
  - soft_contact_loss
  - local_distance_loss
- Stability terms (aux):
  - residual_loss
  - residual_reg
  - coordination_reg
  - smoothness_loss

Contact-focused weighting (default):
- d_gt < tau_contact -> 1.0
- tau_contact <= d_gt < tau_near -> 0.5
- else -> 0.1

## 7. Gated + Bounded Residual

- gate = sigmoid(gate_logits) per joint
- bounded residual via tanh trust region (delta_max)
- no hard gating, no binarization

## 8. Notes and Limitations

- No mesh / SDF / dense vertex contact in v3.
- Default focus remains on upper-body and hands.
- v1/v2 remain unchanged and compatible.


## 9. V3-lite Training Objective (Default)

当前默认训练目标简化为：

- loss_soft (主物理项)
- loss_res (残差稳定器)
- loss_smooth (时间平滑)

默认关闭（作为后续消融开关保留）：
- loss_dist
- loss_local
- loss_reg
- loss_coord

这些项可通过对应 lambda > 0 再打开做 ablation。
