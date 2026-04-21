import argparse
import os
import pickle

import numpy as np

"""
Build a 4-part SMPL-X vertex segmentation for framework attention figures.

This does not replace the existing 6-part segmentation. It uses the same
dominant-joint assignment rule and merges left/right arms and hands. Wrists
are assigned to hands so the highlighted hand region covers the full hand.
"""

PART_JOINT_IDS = {
    "torso_head": [0, 3, 6, 9, 12, 15, 22, 23, 24, 55],
    "lower_body": [1, 2, 4, 5, 7, 8, 10, 11],
    "arms": [13, 14, 16, 17, 18, 19],
    "hands": [20, 21] + list(range(25, 55)),
}


def load_weights(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    return data["weights"]


def build_part_indices(weights):
    num_joints = weights.shape[1]
    part_ids = {name: [i for i in ids if 0 <= i < num_joints] for name, ids in PART_JOINT_IDS.items()}

    vertex_to_joint = np.argmax(weights, axis=1)
    segm = {}
    assigned = np.zeros(weights.shape[0], dtype=bool)
    for name, ids in part_ids.items():
        verts = np.where(np.isin(vertex_to_joint, ids))[0]
        segm[name] = verts.astype(np.int64).tolist()
        assigned[verts] = True

    if not np.all(assigned):
        leftover = np.where(~assigned)[0].astype(np.int64).tolist()
        segm["torso_head"] = segm["torso_head"] + leftover

    return segm


def main():
    base_dir = os.path.dirname(__file__)
    repo_root = os.path.abspath(os.path.join(base_dir, "..", "..", "..", ".."))
    parser = argparse.ArgumentParser(description="Generate 4-part segmentation for SMPL-X.")
    parser.add_argument(
        "--model",
        default=os.path.join(repo_root, "body_models", "smplx", "SMPLX_NEUTRAL.npz"),
        help="Path to SMPL-X model .npz file",
    )
    parser.add_argument(
        "--out",
        default="four_parts.pkl",
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
        out_path = os.path.join(base_dir, out_path)
    with open(out_path, "wb") as f:
        pickle.dump(segm, f)


if __name__ == "__main__":
    main()
