import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch

from stage2_old.proposal.model.refiner_inputs import ContactWindowSampler
from stage2_old.proposal.model.refiner_model import HandContactRefiner
from stage2_old.proposal.model.refiner_loss import HandContactRefinerLoss
from stage2_old.common.geometry.contact_defs import TARGET_PARTS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--frames", type=int, default=12)
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(0)

    batch = 2
    joints = 56
    frames = args.frames

    actor_motion = torch.randn(batch, joints, 6, frames, device=device)
    coarse_motion = torch.randn(batch, joints, 6, frames, device=device)
    gt_motion = torch.randn(batch, joints, 6, frames, device=device)
    lengths = torch.full((batch,), frames, device=device, dtype=torch.long)

    windows = [
        [{"hand_side": "left", "start_frame": 0, "end_frame": 7, "target_part": "actor_left_hand", "target_part_id": 1}],
        [{"hand_side": "right", "start_frame": 2, "end_frame": 9, "target_part": "actor_right_arm", "target_part_id": 4}],
    ]

    frame_labels = {
        "active": torch.zeros(batch, frames, 2, device=device, dtype=torch.long),
        "target_part": torch.zeros(batch, frames, 2, device=device, dtype=torch.long),
        "band": torch.zeros(batch, frames, 2, device=device, dtype=torch.long),
        "phase": torch.zeros(batch, frames, 2, device=device, dtype=torch.long),
    }
    frame_labels["active"][0, :8, 0] = 1
    frame_labels["target_part"][0, :8, 0] = TARGET_PARTS.index("actor_left_hand")
    frame_labels["band"][0, :8, 0] = 1
    frame_labels["phase"][0, :8, 0] = 1

    frame_labels["active"][1, 2:10, 1] = 1
    frame_labels["target_part"][1, 2:10, 1] = TARGET_PARTS.index("actor_right_arm")
    frame_labels["band"][1, 2:10, 1] = 2
    frame_labels["phase"][1, 2:10, 1] = 2

    sampler = ContactWindowSampler(window_size=10, window_pad=0, include_buffer=True, device=device)
    window_batch = sampler.build_window_batch(actor_motion, coarse_motion, gt_motion, lengths, windows, frame_labels)

    model = HandContactRefiner(hidden_dim=128, num_temporal_blocks=2, num_cross_blocks=2, num_spatial_blocks=1)
    model.to(device)
    delta = model(
        window_batch["coarse_local"],
        window_batch["actor_patch_feat"],
        window_batch["relation_feat"],
        window_batch["cond_feat"],
        time_mask=window_batch["time_mask"],
        actor_patch_mask=window_batch["actor_patch_mask"],
    )
    print("delta", tuple(delta.shape))

    joint_ids_t = torch.as_tensor(window_batch["joint_ids"], device=device, dtype=torch.long)
    delta_full = torch.zeros_like(window_batch["coarse_full"])
    delta_full.index_copy_(1, joint_ids_t, delta.permute(0, 2, 3, 1))
    refined_full = window_batch["coarse_full"] + delta_full

    loss_fn = HandContactRefinerLoss()
    loss, loss_dict = loss_fn(
        refined_full,
        window_batch["coarse_full"],
        window_batch["gt_full"],
        window_batch["actor_full"],
        window_batch,
    )
    print("loss_total", loss.detach().item())
    for key, value in loss_dict.items():
        print(key, value.detach().item())


if __name__ == "__main__":
    main()
