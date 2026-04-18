import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch

from stage2_old.proposal.model.proposal_features import HandContactFeatureBuilder
from stage2_old.common.geometry.proposal_labels import HandContactLabelBuilder
from stage2_old.proposal.model.proposal_model import HandContactProposal
from stage2_old.proposal.model.proposal_loss import HandContactProposalLoss
from stage2_old.proposal.model.proposal_events import parse_contact_events
from stage2_old.proposal.model.proposal_windows import ContactWindowBuilder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--frames", type=int, default=20)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--pose_rep", type=str, default="rot6d")
    args = parser.parse_args()

    torch.manual_seed(0)
    device = torch.device(args.device)
    batch = args.batch
    frames = args.frames
    joints = 56

    feat_builder = HandContactFeatureBuilder(device=device, pose_rep=args.pose_rep)
    label_builder = HandContactLabelBuilder(device=device, pose_rep=args.pose_rep)

    if args.pose_rep == "xyz":
        actor_motion = torch.randn(batch, joints, 3, frames, device=device)
        coarse_motion = torch.randn(batch, joints, 3, frames, device=device)
        gt_motion = torch.randn(batch, joints, 3, frames, device=device)
    else:
        actor_motion = torch.randn(batch, joints, 6, frames, device=device)
        coarse_motion = torch.randn(batch, joints, 6, frames, device=device)
        gt_motion = torch.randn(batch, joints, 6, frames, device=device)

    lengths = torch.as_tensor([frames, max(1, frames - 5)], device=device)

    hand_feat, part_feat, rel_feat = feat_builder.build(actor_motion, coarse_motion, lengths=lengths)
    print("hand_feat", tuple(hand_feat.shape))
    print("part_feat", tuple(part_feat.shape))
    print("rel_feat", tuple(rel_feat.shape))

    labels = label_builder.build(actor_motion, gt_motion, lengths=lengths)
    for key, value in labels.items():
        print(key, tuple(value.shape))

    model = HandContactProposal(hand_feat.shape[-1], part_feat.shape[-1]).to(device)
    logits = model(hand_feat, part_feat, rel_feat)
    for key, value in logits.items():
        print(f"logits_{key}", tuple(value.shape))

    criterion = HandContactProposalLoss()
    loss, loss_dict = criterion(logits, labels, lengths=lengths)
    print("loss_total", loss.detach().item())
    for key, value in loss_dict.items():
        print(key, value.detach().item())

    events = parse_contact_events(
        logits["active"],
        logits["target"],
        logits["band"],
        logits["phase"],
        lengths=lengths,
    )
    print("events_per_batch", [len(x) for x in events])

    window_builder = ContactWindowBuilder(window_size=8, pad=2)
    windows = window_builder.build(events, lengths=lengths)
    print("windows_per_batch", [len(x) for x in windows])
    mask = window_builder.to_mask(windows, lengths=lengths, num_hands=2)
    print("window_mask", tuple(mask.shape))


if __name__ == "__main__":
    main()
