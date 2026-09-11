"""Hard routing and truth-gated diagnostic; ties use torch's first argmax."""

import torch


def hard_route(stage1_logits, stage2_logits):
    if (
        stage1_logits.ndim != 2
        or stage1_logits.shape[1] != 2
        or stage2_logits.shape != (len(stage1_logits), 3)
    ):
        raise ValueError("Expected aligned Nx2 and Nx3 logits")
    return torch.where(stage1_logits.argmax(1) == 0, 0, stage2_logits.argmax(1) + 1)


def oracle_route(endpoint_truth, stage2_logits):
    if (
        endpoint_truth.ndim != 1
        or stage2_logits.shape != (len(endpoint_truth), 3)
        or torch.any((endpoint_truth < 0) | (endpoint_truth > 3))
    ):
        raise ValueError("Invalid oracle inputs")
    return torch.where(endpoint_truth == 0, 0, stage2_logits.argmax(1) + 1)


def soft_probabilities(stage1_logits, stage2_logits):
    """P(NM), P(M)P(MEL|M), P(M)P(BCC|M), P(M)P(SCC|M)."""
    if (
        stage1_logits.ndim != 2
        or stage1_logits.shape[1] != 2
        or stage2_logits.shape != (len(stage1_logits), 3)
    ):
        raise ValueError("Expected aligned Nx2 and Nx3 logits")
    p1, p2 = stage1_logits.softmax(1), stage2_logits.softmax(1)
    return torch.cat((p1[:, :1], p1[:, 1:] * p2), dim=1)


def endpoint_output(output, system_type):
    """Return deployable prediction and normalized endpoint probabilities."""
    if system_type == "flat":
        return output.argmax(1), output.softmax(1)
    s1, s2 = output["task1"], output["task2"]
    if system_type in ("shared_soft", "dedicated_soft"):
        probabilities = soft_probabilities(s1, s2)
        return probabilities.argmax(1), probabilities
    if system_type not in ("shared_hard", "dedicated_hard"):
        raise ValueError("Unknown system")
    malignant = (s1.argmax(1) == 1).to(s1.dtype)[:, None]
    probabilities = torch.cat((1 - malignant, malignant * s2.softmax(1)), dim=1)
    return hard_route(s1, s2), probabilities
