import argparse
import os
import pickle

import numpy as np

"""
Based on SMPL-X skinning weights, each mesh vertex is mapped to 
one of your predefined 6 joint-part groups according to its "dominant joint", 
thus obtaining a vertex segmentation for viewer shading.

6-part segmentation:
torso_head, lower_body, left_arm, right_arm, left_hand, right_hand.
"""

PART_JOINT_IDS = {
    "torso_head": [0, 3, 6, 9, 12, 15, 22, 23, 24, 55],
    "lower_body": [1, 2, 4, 5, 7, 8, 10, 11],
    "left_arm": [13, 16, 18, 20],
    "right_arm": [14, 17, 19, 21],
    "left_hand": [25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39],
    "right_hand": [40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54],
}


def load_weights(npz_path):
    # The value represents the strength at which the v-th vertex is controlled by the j-th joint.
    data = np.load(npz_path, allow_pickle=True)
    return data["weights"] # [N_v, N_j] 


def build_part_indices(weights):
    """
    Input:  weights, [N_v, N_j]
    Output: segm, dict[str, list[int]]
    """
    num_joints = weights.shape[1]
    part_ids = {name: [i for i in ids if 0 <= i < num_joints] for name, ids in PART_JOINT_IDS.items()}

    # For each vertex v, select the joint with the largest weight.
    vertex_to_joint = np.argmax(weights, axis=1)
    segm = {}
    assigned = np.zeros(weights.shape[0], dtype=bool)
    for name, ids in part_ids.items():
        verts = np.where(np.isin(vertex_to_joint, ids))[0]
        segm[name] = verts.astype(np.int64).tolist()
        assigned[verts] = True

    # Any leftover vertices are added to torso_head to keep a full partition.
    if not np.all(assigned):
        leftover = np.where(~assigned)[0].astype(np.int64).tolist()
        segm["torso_head"] = segm["torso_head"] + leftover

    return segm


def main():
    base_dir = os.path.dirname(__file__)
    parser = argparse.ArgumentParser(description="Generate 6-part segmentation for SMPL-X.")
    parser.add_argument(
        "--model",
        default=os.path.join(base_dir, "..", "..", "body_models", "smplx", "SMPLX_NEUTRAL.npz"),
        help="Path to SMPL-X model .npz file",
    )
    parser.add_argument(
        "--out",
        default="six_parts.pkl",
        help="Output .pkl path (part_name -> vertex indices)",
    )
    args = parser.parse_args()

    model_path = args.model
    if not os.path.isabs(model_path):
        model_path = os.path.normpath(os.path.join(base_dir, model_path))
    weights = load_weights(model_path)
    segm = build_part_indices(weights)

    out_path = args.out
    if not os.path.isabs(out_path):
        out_path = os.path.join(os.path.dirname(__file__), out_path)
    with open(out_path, "wb") as f:
        pickle.dump(segm, f)



if __name__ == "__main__":
    main()
