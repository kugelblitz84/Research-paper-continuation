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
