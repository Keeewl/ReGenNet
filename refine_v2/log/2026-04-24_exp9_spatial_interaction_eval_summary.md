# exp9 spatial interaction evaluation summary

## Scope

Experiment:

- `refiner_v2_exp9_spatial_interaction_v1_10k`

Goal:

- test whether stronger but still lightweight task-specific spatial interaction
  can improve Stage2 contact refinement beyond exp8

Compared against:

- `exp8_interaction_v1_10k`

## Result summary

Conclusion:

- exp9 does **not** outperform exp8
- exp8 remains the best Stage2 result
- exp9 should be treated as a final no-gain / negative ablation for stronger spatial interaction

## Window-level results

exp9:

- `gt_contact_contact_dist_improvement = 0.0029010999687671232`
- `refined_contact_f1 = 0.818623481298672`
- `topk_refined_contact_f1 = 0.8264006654633981`
- `all_valid_dist_l1_improvement = 0.0022487042122579922`
- `contact_motion_improvement = 0.0021570559828680857`
- `motion_improvement = 0.002019536037316524`

exp8 reference:

- `gt_contact_contact_dist_improvement = 0.003047680515683272`
- `refined_contact_f1 = 0.8214249439940374`
- `topk_refined_contact_f1 = 0.8288972412102136`
- `all_valid_dist_l1_improvement = 0.0025347111161972244`
- `contact_motion_improvement = 0.002881524312023031`
- `motion_improvement = 0.002696233594414481`

Window-level judgment:

- exp9 is weaker than exp8 on all main contact metrics
- stronger spatial interaction did not convert into better local contact refinement

## Full-sequence results

exp9:

- `refined_contact_f1 = 0.7926315553499481`
- `gt_contact_contact_dist_improvement = 0.0016739577986299992`
- `surrogate_penetration_depth_gap_improvement = 0.00023515056818723679`

exp8 reference:

- `refined_contact_f1 = 0.7945530333590035`
- `gt_contact_contact_dist_improvement = 0.001841994933784008`
- `surrogate_penetration_depth_gap_improvement = 0.00024501560255885124`

Full-sequence judgment:

- exp9 is also weaker than exp8 at the system level
- no gain appears after window stitching / full-sequence evaluation either

## Behavior diagnosis

The main pattern is increased conservatism.

exp9 deltas:

- `delta_norm_selected_hand = 0.00992197348203707`
- `delta_norm_same_side_arm = 0.012694433957707054`
- `delta_norm_other_hand_arm = 0.008702964145772447`
- `delta_norm_transl = 0.0001323255160310816`

exp8 deltas:

- `delta_norm_selected_hand = 0.011754125936525648`
- `delta_norm_same_side_arm = 0.015009680076366064`
- `delta_norm_other_hand_arm = 0.009932678241565865`
- `delta_norm_transl = 0.00020620567090465396`

Interpretation:

- exp9 produces smaller effective residuals
- selected-hand / same-side-arm correction is weaker than exp8
- likely causes:
  - stronger interaction block did not improve useful contact alignment
  - added `lambda_gt_relative_overclose = 0.01` further pushed the model to a more conservative solution

## Final conclusion

The last model-side upgrade attempt is complete.

- exp8 remains the best Stage2 model
- exp9 shows that increasing hand-target spatial interaction complexity beyond exp8
  does not automatically improve contact quality
- Stage2 model line can now be treated as basically converged

## Recommended decision

Freeze Stage2 main result as:

- `refiner_v2_exp8_interaction_v1_10k`

Use exp9 only as:

- final no-gain ablation
- evidence that stronger spatial interaction did not improve the current framework
