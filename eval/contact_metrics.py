import torch


def compute_contact_metrics_stub(refined_motion, actor_motion, lengths=None):
    """
    Placeholder for contact metrics.
    Returns a dict with keys for future evaluation.
    """
    device = refined_motion.device
    return {
        "hand_cd": torch.tensor(0.0, device=device),
        "contact_ratio": torch.tensor(0.0, device=device),
        "avg_contact_duration": torch.tensor(0.0, device=device),
        "contact_frequency": torch.tensor(0.0, device=device),
    }
