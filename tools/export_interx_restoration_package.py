import argparse
import os
import pickle

import numpy as np
from tqdm import tqdm


def _normalize_gender_id(value):
    text = str(value).strip().lower()
    if text in {"male", "m"}:
        return 1
    if text in {"female", "f"}:
        return 2
    return 0


def _load_interaction_order(path):
    if not path:
        return {}
    with open(path, "rb") as f:
        return pickle.load(f)


def _object_array(values):
    arr = np.empty((len(values),), dtype=object)
    for i, value in enumerate(values):
        arr[i] = value
    return arr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_motions_root", required=True, type=str)
    parser.add_argument("--interaction_order", required=True, type=str)
    parser.add_argument("--output_path", required=True, type=str)
    parser.add_argument("--downsample", default=4, type=int)
    parser.add_argument("--processed_fps", default=30, type=int)
    parser.add_argument("--limit", default=-1, type=int)
    args = parser.parse_args()

    raw_root = os.path.abspath(args.raw_motions_root)
    order_dict = _load_interaction_order(args.interaction_order)

    clip_names = [
        name for name in sorted(os.listdir(raw_root))
        if os.path.isdir(os.path.join(raw_root, name))
    ]
    if args.limit > 0:
        clip_names = clip_names[: args.limit]

    dataset_key = []
    actor_is_p1 = []
    p1_betas = []
    p2_betas = []
    p1_gender_id = []
    p2_gender_id = []
    p1_trans = []
    p2_trans = []
    p1_root_orient = []
    p2_root_orient = []
    raw_nframes = []
    downsample = []
    processed_fps = []
    raw_fps = []

    for clip_name in tqdm(clip_names, desc="Export Inter-X restoration package"):
        p1_path = os.path.join(raw_root, clip_name, "P1.npz")
        p2_path = os.path.join(raw_root, clip_name, "P2.npz")
        if not os.path.exists(p1_path) or not os.path.exists(p2_path):
            raise FileNotFoundError(f"Missing raw clip files: {p1_path} / {p2_path}")

        p1 = np.load(p1_path, allow_pickle=True)
        p2 = np.load(p2_path, allow_pickle=True)

        dataset_key.append(clip_name)
        actor_is_p1.append(1 if int(order_dict.get(clip_name, 1)) == 1 else 0)
        p1_betas.append(np.asarray(p1["betas"], dtype=np.float32).reshape(-1))
        p2_betas.append(np.asarray(p2["betas"], dtype=np.float32).reshape(-1))
        p1_gender_id.append(_normalize_gender_id(p1["gender"]))
        p2_gender_id.append(_normalize_gender_id(p2["gender"]))
        p1_trans.append(np.asarray(p1["trans"], dtype=np.float32))
        p2_trans.append(np.asarray(p2["trans"], dtype=np.float32))
        p1_root_orient.append(np.asarray(p1["root_orient"], dtype=np.float32))
        p2_root_orient.append(np.asarray(p2["root_orient"], dtype=np.float32))
        raw_nframes.append(int(np.asarray(p1["trans"]).shape[0]))
        downsample.append(int(args.downsample))
        processed_fps.append(int(args.processed_fps))
        raw_fps.append(int(args.processed_fps) * int(args.downsample))

    payload = {
        "dataset_key": np.asarray(dataset_key, dtype=object),
        "actor_is_p1": np.asarray(actor_is_p1, dtype=np.int64),
        "p1_betas": np.stack(p1_betas, axis=0).astype(np.float32),
        "p2_betas": np.stack(p2_betas, axis=0).astype(np.float32),
        "p1_gender_id": np.asarray(p1_gender_id, dtype=np.int64),
        "p2_gender_id": np.asarray(p2_gender_id, dtype=np.int64),
        "p1_trans": _object_array(p1_trans),
        "p2_trans": _object_array(p2_trans),
        "p1_root_orient": _object_array(p1_root_orient),
        "p2_root_orient": _object_array(p2_root_orient),
        "raw_nframes": np.asarray(raw_nframes, dtype=np.int64),
        "downsample": np.asarray(downsample, dtype=np.int64),
        "processed_fps": np.asarray(processed_fps, dtype=np.int64),
        "raw_fps": np.asarray(raw_fps, dtype=np.int64),
    }

    output_path = os.path.abspath(args.output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez_compressed(output_path, **payload)
    print(f"Saved restoration package to {output_path} (clips={len(dataset_key)})")


if __name__ == "__main__":
    main()
