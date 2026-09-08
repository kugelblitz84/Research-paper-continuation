"""Single encoder factory; keep native classifier initialization order."""

from dataclasses import dataclass

import torch
from torch import nn
from torchvision import models

from src.config import ARCHITECTURES

WEIGHT_ENUMS = dict(
    zip(
        ARCHITECTURES,
        (
            "EfficientNet_B0_Weights",
            "DenseNet121_Weights",
            "DenseNet169_Weights",
            "ResNet50_Weights",
            "MobileNet_V3_Large_Weights",
            "EfficientNet_B2_Weights",
            "EfficientNet_B3_Weights",
        ),
    )
)


@dataclass
class EncoderParts:
    encoder: nn.Module
    pool: nn.Module
    dimension: int
    flat_projection: nn.Module
    weight_name: str


class DenseFeatures(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.features = features

    def forward(self, x):
        return torch.nn.functional.relu(self.features(x), inplace=True)


def build_encoder(architecture, pretrained=True):
    if architecture not in ARCHITECTURES:
        raise ValueError(f"Unsupported backbone: {architecture}")
    weights = getattr(models, WEIGHT_ENUMS[architecture]).DEFAULT if pretrained else None
    native = getattr(models, architecture)(weights=weights)
    projection = nn.Identity()
    if architecture.startswith("densenet"):
        encoder, pool, dim = (
            DenseFeatures(native.features),
            nn.AdaptiveAvgPool2d(1),
            native.classifier.in_features,
        )
    elif architecture == "resnet50":
        encoder, pool, dim = (
            nn.Sequential(*list(native.children())[:-2]),
            native.avgpool,
            native.fc.in_features,
        )
    elif architecture == "mobilenet_v3_large":
        encoder, pool, dim = (
            native.features,
            native.avgpool,
            native.classifier[0].in_features,
        )
        # Flat historical builder retained the pretrained projection and Hardswish.
        projection = nn.Sequential(native.classifier[0], native.classifier[1])
    else:
        encoder, pool, dim = (
            native.features,
            native.avgpool,
            native.classifier[-1].in_features,
        )
    return EncoderParts(encoder, pool, dim, projection, str(weights))
