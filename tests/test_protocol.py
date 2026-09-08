import pytest
import torch
from PIL import Image

from src.config import ARCHITECTURES, CLASSES, ROOT, load_config
from src.data import audit_manifests, encode_targets, read_manifest
from src.losses import MaskedLoss
from src.models.systems import Flat, SharedHard
from src.routing import hard_route, oracle_route
from src.transforms import build_transform


@pytest.mark.parametrize("system", ["flat", "shared_hard"])
@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_configs_and_all_model_outputs(system, architecture):
    c = load_config(f"configs/experiments/{system}/{architecture}.yaml")
    from src.models import build_model

    model = build_model(c, pretrained=False).eval()
    calls = []
    hook = model.encoder.register_forward_hook(lambda *args: calls.append(1))
    with torch.inference_mode():
        output = model(torch.randn(1, 3, 224, 224))
    assert calls == [1]
    hook.remove()
    if system == "flat":
        assert output.shape == (1, 4)
    else:
        assert [output[f"task{i}"].shape for i in (1, 2, 3)] == [(1, 2), (1, 3), (1, 5)]
        assert sum(k == "encoder" for k, _ in model.named_children()) == 1
        assert model.task1_head[0].p == model.task2_head[0].p == model.task3_head[0].p == 0.2


def test_manifest_integrity_order_disjointness(config):
    audit = audit_manifests(config)
    assert audit["isic"]["counts"] == {"train": 17124, "validation": 3668, "test": 3668}
    assert not audit["cross_source_split_conflicts"]
    assert CLASSES == ("non_malignant", "melanoma", "bcc", "scc")
    frame = read_manifest(config, "isic")
    assert not (frame.diagnosis_canonical == "actinic_keratosis").any()
    assert frame.loc[frame.split == "test"].target.value_counts().sort_index().tolist() == [
        2398,
        678,
        498,
        94,
    ]


def test_manifest_tamper_rejected(config, tmp_path, monkeypatch):
    import shutil

    import src.data as d

    (tmp_path / "data/manifests").mkdir(parents=True)
    for p in (ROOT / "data/manifests").glob("*"):
        shutil.copy(p, tmp_path / "data/manifests" / p.name)
    path = tmp_path / config["manifests"]["isic"]
    with path.open("ab") as f:
        f.write(b"\n")
    monkeypatch.setattr(d, "ROOT", tmp_path)
    with pytest.raises(ValueError, match="hash mismatch"):
        d.read_manifest(config, "isic")


def test_eval_transform_deterministic(config):
    transform = build_transform(config)
    image = Image.effect_noise((301, 263), 80).convert("RGB")
    a, b = transform(image), transform(image)
    assert torch.equal(a, b) and a.shape == (3, 224, 224)
    assert a.dtype == torch.float32


def test_routing_and_oracle():
    s1 = torch.tensor([[3.0, 1.0], [1.0, 3.0], [0.0, 0.0], [2.0, 1.0]])
    s2 = torch.tensor([[0.0, 2.0, 1.0], [0.0, 0.0, 3.0], [3.0, 1.0, 2.0], [0.0, 3.0, 1.0]])
    y = torch.tensor([0, 3, 1, 2])
    assert hard_route(s1, s2).tolist() == [0, 3, 0, 0]
    assert oracle_route(y, s2).tolist() == [0, 3, 1, 2]
    with pytest.raises(ValueError):
        oracle_route(torch.tensor([-1]), s2[:1])


def test_targets_masking_and_absent_task(config):
    targets, masks = zip(*(encode_targets(i) for i in range(4)))
    targets, masks = torch.stack(targets), torch.stack(masks)
    assert targets.tolist() == [
        [0, -100, -100],
        [1, 0, -100],
        [1, 1, -100],
        [1, 2, -100],
    ]
    assert encode_targets(4, "stage3")[0].tolist() == [-100, -100, 4]
    logits = {
        f"task{i + 1}": torch.zeros(4, n, requires_grad=True) for i, n in enumerate((2, 3, 5))
    }
    criterion = MaskedLoss(config)
    total, parts, counts = criterion(logits, targets, masks)
    assert counts == {"task1": 4, "task2": 3, "task3": 0}
    expected = (
        torch.log(torch.tensor(2.0))
        + torch.tensor(config["losses"]["task2_weights"]).mean()
        * (2 / 3) ** 2
        * torch.log(torch.tensor(3.0))
    ) / 2
    assert torch.allclose(total, expected)
    total.backward()
    assert logits["task3"].grad is None
    assert torch.equal(logits["task2"].grad[0], torch.zeros(3))
    with pytest.raises(ValueError):
        criterion(logits, targets, torch.zeros_like(masks))
    targets[0, 0] = -100
    with pytest.raises(ValueError):
        criterion(logits, targets, masks)


def test_invalid_models():
    with pytest.raises(ValueError):
        Flat("resnet18", False)


def test_training_api_cannot_construct_test(config, monkeypatch):
    import src.data as d

    seen = []

    class Fake:
        def __init__(self, c, source, split, *args, **kwargs):
            import pandas as pd

            seen.append((source, split))
            self.frame = pd.DataFrame({"target": [0, 1]})

        def __len__(self):
            return 2

    monkeypatch.setattr(d, "SkinDataset", Fake)
    d.training_loaders(config, ".", workers=0)
    assert seen == [
        ("isic", "train"),
        ("isic", "validation"),
        ("stage3", "train"),
        ("stage3", "validation"),
    ]


@pytest.mark.parametrize("system", ["flat", "shared_hard"])
def test_actual_b0_backward_smoke(config, system):
    from src.training import optimizer_components, training_epoch

    config["system_type"] = system
    model = (Flat if system == "flat" else SharedHard)("efficientnet_b0", False)
    device = torch.device("cpu")
    optimizer, scheduler, scaler = optimizer_components(model, config, device)
    values = [encode_targets(0), encode_targets(2), encode_targets(4, "stage3")]
    batch = dict(
        image=torch.randn(3, 3, 64, 64),
        target=torch.tensor([0, 2, 3]),
        targets=torch.stack([a for a, b in values]),
        masks=torch.stack([b for a, b in values]),
    )
    before = next(model.parameters()).detach().clone()
    result = training_epoch(model, [batch], config, device, optimizer, scaler, MaskedLoss(config))
    assert result["train_loss"] > 0
    assert not torch.equal(before, next(model.parameters()))
    if system == "shared_hard":
        assert all(result[f"train_task{i}_loss"] is not None for i in (1, 2, 3))
