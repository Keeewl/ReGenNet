"""Schema constants for refine_v2 refiner window samples."""

from __future__ import annotations


TENSOR_KEYS = (
    "actor_motion_window",
    "coarse_motion_window",
    "gt_motion_window",
    "coarse_region_contact_mask_window",
    "coarse_min_region_dist_window",
    "gt_region_contact_mask_window",
    "gt_min_region_dist_window",
    "valid_mask",
    "topk_target_region_ids",
    "topk_region_scores_numeric",
    "primary_relative_vector_window",
    "primary_relative_dist_window",
    "topk_relative_vectors_window",
    "topk_relative_dists_window",
)

INT_TENSOR_KEYS = (
    "hand_side_id",
    "primary_target_region_id",
    "window_length",
    "start_frame",
    "end_frame",
    "raw_start_frame",
    "raw_end_frame",
    "dataset_row_index",
    "sample_index",
    "window_index",
    "sequence_window_index",
)

METADATA_KEYS = (
    "dataset_key",
    "action_type",
    "action_label",
    "action_name",
    "bucket_label",
    "hand_side",
    "primary_target_region",
    "topk_target_regions",
    "topk_region_scores",
    "region_score_table",
    "is_gt_positive",
    "is_pred_positive",
)

TOPK_SCORE_FIELDS = (
    "num_contact_frames",
    "mean_min_dist",
    "min_dist",
)
