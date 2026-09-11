"""Exact historical focal formula and masked active-task normalization."""

import torch
from torch import nn
from torch.nn import functional as F


class MaskedLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.register_buffer(
            "task2_weights",
            torch.tensor(config["losses"]["task2_weights"], dtype=torch.float32),
        )
        self.register_buffer(
            "task3_weights",
            torch.tensor(config["losses"]["task3_weights"], dtype=torch.float32),
        )
        self.gamma = config["losses"]["gamma"]
        self.weights = config["losses"]["task_weights"]
        self.task_sizes = (2, 3, 5) if config["system_type"] == "shared_hard" else (2, 3)

    def forward(self, logits, targets, masks):
        if (
            targets.ndim != 2
            or targets.shape[1] != 3
            or targets.dtype != torch.long
            or masks.shape != targets.shape
            or masks.dtype != torch.bool
        ):
            raise ValueError("Expected long Nx3 targets and boolean masks")
        losses, counts, weighted, denominator = {}, {}, [], 0.0
        for i, size in enumerate(self.task_sizes):
            key = f"task{i + 1}"
            if logits[key].shape != (len(targets), size):
                raise ValueError("Wrong task output dimensions")
            active = masks[:, i]
            counts[key] = int(active.sum())
            losses[key] = None
            if not counts[key]:
                continue
            y, x = targets[active, i], logits[key][active]
            if torch.any((y < 0) | (y >= size)):
                raise ValueError("Invalid active target")
            if i == 1:
                logp = F.log_softmax(x, 1).gather(1, y[:, None]).squeeze(1)
                loss = (-self.task2_weights[y] * (1 - logp.exp()).pow(self.gamma) * logp).mean()
            else:
                loss = F.cross_entropy(x, y, weight=self.task3_weights if i == 2 else None)
            losses[key] = loss
            weighted.append(loss * self.weights[i])
            denominator += self.weights[i]
        if not weighted:
            raise ValueError("No active task in batch")
        return torch.stack(weighted).sum() / denominator, losses, counts
