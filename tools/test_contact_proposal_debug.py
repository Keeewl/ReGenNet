import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch

from model.contact.proposal_labels import HandContactLabelBuilder
from model.contact.proposal_events import parse_contact_events
from model.contact.proposal_windows import ContactWindowBuilder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--pose_rep", type=str, default="rot6d")
    args = parser.parse_args()

    torch.manual_seed(0)
    device = torch.device(args.device)
    joints = 56
    frames = args.frames

    if args.pose_rep == "xyz":
        actor_motion = torch.randn(1, joints, 3, frames, device=device)
        gt_motion = torch.randn(1, joints, 3, frames, device=device)
    else:
        actor_motion = torch.randn(1, joints, 6, frames, device=device)
        gt_motion = torch.randn(1, joints, 6, frames, device=device)

    lengths = torch.as_tensor([frames], device=device)

    label_builder = HandContactLabelBuilder(device=device, pose_rep=args.pose_rep)
    labels = label_builder.build(actor_motion, gt_motion, lengths=lengths)

    active = labels["active"]
    target = labels["target_part"]
    band = labels["band"]
    phase = labels["phase"]

    print("left target", target[0, :, 0].tolist())
    print("right target", target[0, :, 1].tolist())
    print("left band", band[0, :, 0].tolist())
    print("right band", band[0, :, 1].tolist())
    print("left phase", phase[0, :, 0].tolist())
    print("right phase", phase[0, :, 1].tolist())

    events = parse_contact_events(active, target, band, phase, lengths=lengths)
    print("events", events)

    window_builder = ContactWindowBuilder(window_size=8, pad=2)
    windows = window_builder.build(events, lengths=lengths)
    print("windows", windows)


if __name__ == "__main__":
    main()
