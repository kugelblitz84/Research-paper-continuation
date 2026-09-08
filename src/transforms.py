"""Historical torchvision-v2 operation order, with config-owned parameters."""

import torch
from torchvision.transforms import InterpolationMode, v2


def build_transform(config, train=False):
    p = config["transforms"]
    bilinear = InterpolationMode.BILINEAR
    ops = [v2.ToImage()]
    if train:
        ops += [
            v2.RandomResizedCrop(
                (p["size"], p["size"]),
                scale=tuple(p["crop_scale"]),
                ratio=tuple(p["crop_ratio"]),
                interpolation=bilinear,
                antialias=True,
            ),
            v2.RandomHorizontalFlip(p["flip"]),
            v2.RandomVerticalFlip(p["flip"]),
            v2.RandomRotation(p["rotation"], interpolation=bilinear),
            v2.ColorJitter(*p["jitter"]),
        ]
    else:
        ops += [
            v2.Resize(p["resize"], interpolation=bilinear, antialias=True),
            v2.CenterCrop((p["size"], p["size"])),
        ]
    return v2.Compose(
        ops + [v2.ToDtype(torch.float32, scale=True), v2.Normalize(p["mean"], p["std"])]
    )
