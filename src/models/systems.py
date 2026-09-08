"""Flat and original shared three-task training with four-class hard endpoints."""

import torch
from torch import nn

from .backbones import build_encoder


class Flat(nn.Module):
    def __init__(self, architecture, pretrained=True, dropout=0.2):
        super().__init__()
        parts = build_encoder(architecture, pretrained)
        self.encoder, self.pool = parts.encoder, parts.pool
        self.feature_dimension, self.weight_name = parts.dimension, parts.weight_name
        self.projection = parts.flat_projection
        dimension = parts.dimension
        if architecture == "mobilenet_v3_large":
            dimension = self.projection[0].out_features
        inplace = architecture.startswith(("efficientnet", "densenet"))
        self.head = nn.Sequential(nn.Dropout(dropout, inplace=inplace), nn.Linear(dimension, 4))

    def forward(self, images):
        return self.head(self.projection(torch.flatten(self.pool(self.encoder(images)), 1)))


class SharedHard(nn.Module):
    def __init__(self, architecture, pretrained=True, dropout=0.2):
        super().__init__()
        parts = build_encoder(architecture, pretrained)
        self.encoder, self.pool = parts.encoder, parts.pool
        self.feature_dimension, self.weight_name = parts.dimension, parts.weight_name
        self.task1_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(parts.dimension, 2))
        self.task2_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(parts.dimension, 3))
        self.task3_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(parts.dimension, 5))

    def forward(self, images):
        vector = torch.flatten(self.pool(self.encoder(images)), 1)
        return {f"task{i}": getattr(self, f"task{i}_head")(vector) for i in (1, 2, 3)}


def build_model(config, *, pretrained=None):
    if config["system_type"] not in ("flat", "shared_hard"):
        raise ValueError("Unknown system")
    use_weights = config["pretrained"] == "imagenet" if pretrained is None else pretrained
    cls = Flat if config["system_type"] == "flat" else SharedHard
    return cls(config["architecture"], use_weights, config["dropout"])
