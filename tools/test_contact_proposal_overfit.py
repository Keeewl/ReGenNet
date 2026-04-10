import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch

from model.contact.proposal_features import HandContactFeatureBuilder
from model.contact.proposal_labels import HandContactLabelBuilder
from model.contact.proposal_model import HandContactProposal
from model.contact.proposal_loss import HandContactProposalLoss
from model.contact.contact_geometry import build_time_mask


def compute_metrics(logits, labels, lengths):
    num_frames = logits["active"].shape[1]
    mask = build_time_mask(lengths, num_frames, device=logits["active"].device)
    if mask is None:
        mask = torch.ones(labels["active"].shape[0], num_frames, device=logits["active"].device, dtype=torch.bool)
    mask = mask[:, :, None]
    mask_hand = mask.expand(-1, -1, logits["active"].shape[2])

    active_pred = torch.sigmoid(logits["active"].squeeze(-1)) > 0.5
    active_gt = labels["active"] > 0.5
    active_mask = mask_hand.bool()

    tp = (active_pred & active_gt & active_mask).sum().float()
    fp = (active_pred & ~active_gt & active_mask).sum().float()
    fn = (~active_pred & active_gt & active_mask).sum().float()
    precision = tp / (tp + fp).clamp(min=1.0)
    recall = tp / (tp + fn).clamp(min=1.0)
    f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-6)

    def acc_from_logits(logit_key, label_key, num_classes):
        pred = torch.argmax(logits[logit_key], dim=-1)
        gt = labels[label_key]
        correct = (pred == gt) & active_mask
        total = active_mask.sum().float().clamp(min=1.0)
        return correct.sum().float() / total

    return {
        "active_precision": float(precision),
        "active_recall": float(recall),
        "active_f1": float(f1),
        "target_acc": float(acc_from_logits("target", "target_part", 6)),
        "band_acc": float(acc_from_logits("band", "band", 3)),
        "phase_acc": float(acc_from_logits("phase", "phase", 4)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--pose_rep", type=str, default="rot6d")
    args = parser.parse_args()

    torch.manual_seed(0)
    device = torch.device(args.device)
    batch = 1
    frames = 30
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

    lengths = torch.as_tensor([frames], device=device)

    hand_feat, part_feat, rel_feat = feat_builder.build(actor_motion, coarse_motion, lengths=lengths)
    labels = label_builder.build(actor_motion, gt_motion, lengths=lengths)

    model = HandContactProposal(hand_feat.shape[-1], part_feat.shape[-1]).to(device)
    criterion = HandContactProposalLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    init_loss = None
    for step in range(args.steps):
        logits = model(hand_feat, part_feat, rel_feat)
        loss, _ = criterion(logits, labels, lengths=lengths)
        if init_loss is None:
            init_loss = loss.detach().item()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 20 == 0 or step == args.steps - 1:
            print(f"step {step} loss {loss.detach().item():.6f}")

    final_logits = model(hand_feat, part_feat, rel_feat)
    final_loss, _ = criterion(final_logits, labels, lengths=lengths)
    metrics = compute_metrics(final_logits, labels, lengths)
    print(f"init_loss {init_loss:.6f}")
    print(f"final_loss {final_loss.detach().item():.6f}")
    for key, value in metrics.items():
        print(f"{key} {value:.4f}")


if __name__ == "__main__":
    main()
