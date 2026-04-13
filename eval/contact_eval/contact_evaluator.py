import torch

from model.contact.contact_geometry import ContactGeometry
from model.contact.proposal_labels import HandContactLabelBuilder

from eval.contact_eval.contact_metrics import (
    compute_hand_cd,
    compute_contact_ratio,
    compute_avg_contact_duration,
    compute_contact_frequency,
    build_contact_labels,
)
from eval.contact_eval.contact_segments import build_union_contact_mask


class HandContactEvaluator:
    def __init__(
        self,
        body_model="smplx",
        pose_rep="rot6d",
        translation=True,
        glob=True,
        tau_contact=0.10,
        tau_near=0.18,
        topk=3,
        device="cpu",
    ):
        self.pose_rep = pose_rep
        self.body_model = body_model
        self.translation = translation
        self.glob = glob
        self.tau_contact = float(tau_contact)
        self.tau_near = float(tau_near)
        self.topk = int(topk)

        self.label_builder = HandContactLabelBuilder(
            body_model=body_model,
            pose_rep=pose_rep,
            translation=translation,
            glob=glob,
            tau_contact=tau_contact,
            tau_near=tau_near,
            topk=topk,
            device=device,
        )
        self.geometry = ContactGeometry(
            body_model=body_model,
            pose_rep=pose_rep,
            translation=translation,
            glob=glob,
            device=device,
        )

    def evaluate(
        self,
        actor_motion,
        reactor_motion,
        lengths=None,
        gt_reactor_motion=None,
        return_debug=False,
    ):
        labels = build_contact_labels(
            actor_motion,
            reactor_motion,
            lengths=lengths,
            label_builder=self.label_builder,
        )
        contact_mask = build_union_contact_mask(labels["band"], lengths=lengths)

        ratio_stats = compute_contact_ratio(contact_mask, lengths=lengths)
        duration_stats = compute_avg_contact_duration(contact_mask, lengths=lengths)
        freq_stats = compute_contact_frequency(contact_mask, lengths=lengths)

        results = {
            "hand_cd": None,
            "contact_ratio": float(ratio_stats["contact_ratio"]),
            "avg_contact_duration": float(duration_stats["avg_contact_duration"]),
            "contact_frequency": float(freq_stats["contact_frequency"]),
            "num_valid_sequences": int(ratio_stats["num_valid_sequences"]),
            "num_contact_segments": int(duration_stats["num_contact_segments"]),
            "num_contact_frames": int(ratio_stats["num_contact_frames"]),
        }

        if gt_reactor_motion is not None:
            cd_stats = compute_hand_cd(
                actor_motion,
                reactor_motion,
                gt_reactor_motion,
                lengths=lengths,
                label_builder=self.label_builder,
                geometry=self.geometry,
                topk=self.topk,
                return_debug=return_debug,
            )
            if cd_stats["hand_cd"] is None:
                results["hand_cd"] = None
            else:
                results["hand_cd"] = float(cd_stats["hand_cd"])

            if return_debug and "hand_cd_topk_mean" in cd_stats:
                results["hand_cd_topk_mean"] = float(cd_stats["hand_cd_topk_mean"])
                results["hand_cd_count"] = float(cd_stats["hand_cd_count"])

        return results
