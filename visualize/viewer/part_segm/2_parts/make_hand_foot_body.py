import argparse
import os
import pickle

import numpy as np

"""
Generate vertex indices for the hand/foot/body triads from the SMPL-X model file.
"""

def load_joint_map(npz_path):
    # Load skinning weights and joint name mapping from the SMPL-X model.
    data = np.load(npz_path, allow_pickle=True)
    joint2num = data["joint2num"]
    if isinstance(joint2num, np.ndarray):
        joint2num = joint2num.item()
    return data["weights"], joint2num


def build_part_indices(weights, joint2num):
    # Group joints by name tokens, then assign each vertex to its max-weight joint.
    hand_tokens = ("Wrist", "Thumb", "Index", "Middle", "Ring", "Pinky", "Hand")
    foot_tokens = ("Ankle", "Foot", "Toes")

    hand_joints = {j for j in joint2num if any(t in j for t in hand_tokens)}
    foot_joints = {j for j in joint2num if any(t in j for t in foot_tokens)}

    hand_ids = [joint2num[j] for j in sorted(hand_joints)]
    foot_ids = [joint2num[j] for j in sorted(foot_joints)]

    vertex_to_joint = np.argmax(weights, axis=1)
    hand_verts = np.where(np.isin(vertex_to_joint, hand_ids))[0]
    foot_verts = np.where(np.isin(vertex_to_joint, foot_ids))[0]

    all_verts = np.arange(weights.shape[0])
    body_mask = ~(np.isin(all_verts, hand_verts) | np.isin(all_verts, foot_verts))
    body_verts = all_verts[body_mask]

    return {
        "hand": hand_verts.astype(np.int64).tolist(),
        "foot": foot_verts.astype(np.int64).tolist(),
        "body": body_verts.astype(np.int64).tolist(),
    }


def main():
    base_dir = os.path.dirname(__file__)
    parser = argparse.ArgumentParser(description="Generate hand/foot/body parts segmentation for SMPL-X.")
    parser.add_argument(
        "--model",
        default=os.path.join(base_dir, "..", "..", "body_models", "smplx", "SMPLX_NEUTRAL.npz"),
        help="Path to SMPL-X model .npz file",
    )
    parser.add_argument(
        "--out",
        default="hand_foot_body.pkl",
        help="Output .pkl path (part_name -> vertex indices)",
    )
    args = parser.parse_args()
    
    # Resolve model path relative to this script.
    model_path = args.model
    if not os.path.isabs(model_path):
        model_path = os.path.normpath(os.path.join(base_dir, model_path))
    weights, joint2num = load_joint_map(model_path)
    segm = build_part_indices(weights, joint2num)

    # Write part segmentation as {part_name: vertex_indices}.
    out_path = args.out
    if not os.path.isabs(out_path):
        out_path = os.path.join(os.path.dirname(__file__), out_path)
    with open(out_path, "wb") as f:
        pickle.dump(segm, f)



if __name__ == "__main__":
    main()
