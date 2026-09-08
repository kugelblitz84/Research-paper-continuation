import pytest
import torch

from src.artifacts import load_checkpoint, save_checkpoint
from src.training import _allow_backbone, train


def test_checkpoint_roundtrip_no_overwrite(tmp_path):
    model = torch.nn.Linear(3, 4)
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 30)
    x = torch.randn(2, 3)
    model(x).sum().backward()
    optimizer.step()
    scheduler.step()
    state = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": 1,
    }
    path = tmp_path / "last.pt"
    ref = save_checkpoint(path, state)
    loaded = load_checkpoint(ref)
    restored = torch.nn.Linear(3, 4)
    restored.load_state_dict(loaded["model_state_dict"])
    assert torch.equal(model(x), restored(x))
    assert loaded["optimizer_state_dict"]["state"]
    with pytest.raises(FileExistsError):
        save_checkpoint(path, state)
    with path.open("ab") as f:
        f.write(b"bad")
    with pytest.raises(ValueError, match="hash"):
        load_checkpoint(ref)


def test_training_disabled_and_non_b0_gate(config):
    assert train(config)["status"] == "not_started"
    config["architecture"] = "densenet121"
    with pytest.raises(RuntimeError, match="locked"):
        _allow_backbone(config)


def test_in_memory_scientific_mutation_blocked(config):
    config["training"]["lr"] = 0.001
    with pytest.raises(ValueError, match="modified in memory"):
        train(config, allow_training=True)
