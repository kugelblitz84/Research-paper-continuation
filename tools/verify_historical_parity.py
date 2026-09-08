"""Optional local audit against separately preserved ZIP source; never a runtime dependency."""

import importlib.util
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from PIL import Image

from src.config import ARCHITECTURES, load_config
from src.models.systems import Flat, SharedHard
from src.transforms import build_transform


def module(name, relative):
    path = ROOT / "reference/historical" / relative
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def main():
    torch.set_num_threads(2)
    b0 = module("old_b0", "src/models/efficientnet_baseline.py")
    dn = module("old_dn", "src/models/densenet_baseline.py")
    others = module("old_others", "src/models/phase02_backbones.py")
    shared = module("old_shared", "src/models/shared_three_task.py")
    old_transforms = module("old_transforms", "src/data/transforms.py")
    results = []
    torch.manual_seed(9)
    x = torch.randn(1, 3, 224, 224)
    for system in ("flat", "shared_hard"):
        for a in ARCHITECTURES:
            torch.manual_seed(42)
            if system == "shared_hard":
                old = shared.build_shared_three_task_model(a, pretrained="none")
            elif a == "efficientnet_b0":
                old = b0.build_efficientnet_b0(4, pretrained="none")
            elif a == "densenet121":
                old = dn.build_densenet121(4, pretrained="none")
            else:
                old = others.build_phase02_backbone(a, 4, pretrained="none")
            torch.manual_seed(42)
            new = (Flat if system == "flat" else SharedHard)(a, False)
            old.eval()
            new.eval()
            with torch.inference_mode():
                y, z = old(x), new(x)
            diffs = (
                [float((y[k] - z[k]).abs().max()) for k in y]
                if isinstance(y, dict)
                else [float((y - z).abs().max())]
            )
            params = sum(p.numel() for p in old.parameters()) == sum(
                p.numel() for p in new.parameters()
            )
            if max(diffs) != 0 or not params:
                raise AssertionError((a, system, diffs, params))
            # Identical train-mode random initialization, dropout and batchnorm behavior.
            old.train()
            new.train()
            torch.manual_seed(71)
            y = old(x.repeat(2, 1, 1, 1))
            torch.manual_seed(71)
            z = new(x.repeat(2, 1, 1, 1))
            train_diffs = (
                [float((y[k] - z[k]).abs().max().detach()) for k in y]
                if isinstance(y, dict)
                else [float((y - z).abs().max().detach())]
            )
            if max(train_diffs) != 0:
                raise AssertionError((a, system, train_diffs))
            results.append(
                dict(
                    architecture=a,
                    system=system,
                    eval_max_abs_difference=max(diffs),
                    train_max_abs_difference=max(train_diffs),
                    parameter_count=sum(p.numel() for p in new.parameters()),
                    status="PASS",
                )
            )
            del old, new, y, z
    c = load_config("configs/experiments/flat/efficientnet_b0.yaml")
    image = Image.effect_noise((301, 277), 90).convert("RGB")
    for train in (False, True):
        old = (
            old_transforms.build_train_transform if train else old_transforms.build_eval_transform
        )()
        torch.manual_seed(52)
        a = old(image)
        torch.manual_seed(52)
        b = build_transform(c, train)(image)
        assert torch.equal(a, b)
    result = dict(
        model_parity=results,
        transforms_train_eval="bitwise_equal",
        weights="none; identical seed; no pretrained download",
        scope="CPU forward parity, not real training reproduction",
    )
    (ROOT / "validation/historical_parity.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
