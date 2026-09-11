"""Stable Block-A identities; the encoder factory owns supported backbones."""

from src.config import ARCHITECTURES, load_config

SYSTEMS = {
    "flat": ("F", "Flat"),
    "shared_hard": ("H1", "H1_Shared_Hard"),
    "shared_soft": ("H2", "H2_Shared_Soft"),
    "dedicated_hard": ("H3", "H3_Dedicated_Hard"),
    "dedicated_soft": ("H4", "H4_Dedicated_Soft"),
}
BACKBONE_LABELS = {
    "densenet121": "DenseNet121",
    "densenet169": "DenseNet169",
    "resnet50": "ResNet50",
    "mobilenet_v3_large": "MobileNetV3-Large",
    "efficientnet_b0": "EfficientNet-B0",
    "efficientnet_b2": "EfficientNet-B2",
    "efficientnet_b3": "EfficientNet-B3",
}
assert set(BACKBONE_LABELS) == set(ARCHITECTURES)


def experiment_config(system, backbone, seed=42):
    if system not in SYSTEMS or backbone not in ARCHITECTURES:
        raise ValueError("Unsupported system/backbone")
    config = load_config(f"configs/experiments/{system}/{backbone}.yaml")
    if seed != config["seed"]:
        raise ValueError("Seed differs from frozen protocol; explicit protocol version required")
    return config


def expected_cells(seed=42):
    return [
        dict(
            system_type=s,
            system_code=SYSTEMS[s][0],
            backbone=b,
            seed=seed,
            run_id=f"{s}_{b}_seed{seed}",
        )
        for b in BACKBONE_LABELS
        for s in SYSTEMS
    ]
