"""Tiny synthetic end-to-end runs exercise production orchestration without real images/GPU."""

import copy
import json
import shutil

import pandas as pd
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src import artifacts, evaluation, statistics, training
from src.config import ROOT


class TinyDataset(Dataset):
    def __init__(self, config=None, source="isic", split="train", root=None, **kwargs):
        self.source = source
        self.split = split
        self.frame = pd.DataFrame(
            dict(image_id=[f"image_{i}" for i in range(8)], target=[0, 1, 2, 3] * 2)
        )
        if source == "stage3":
            self.frame["target"] = [0, 1, 2, 3, 4, 0, 1, 2]
        self.images = torch.arange(8 * 3 * 8 * 8, dtype=torch.float32).reshape(8, 3, 8, 8) / 1000

    def __len__(self):
        return 8

    def __getitem__(self, i):
        from src.data import encode_targets

        y = int(self.frame.target.iloc[i])
        targets, masks = encode_targets(y, self.source)
        return dict(
            image=self.images[i],
            target=torch.tensor(y),
            targets=targets,
            masks=masks,
            image_id=self.frame.image_id.iloc[i],
        )


class TinyModel(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.system = config["system_type"]
        self.encoder = nn.Sequential(nn.Flatten(), nn.Linear(192, 8), nn.ReLU(), nn.Dropout(0.2))
        if self.system == "flat":
            self.head = nn.Linear(8, 4)
        else:
            self.heads = nn.ModuleList([nn.Linear(8, n) for n in (2, 3, 5)])

    def forward(self, x):
        x = self.encoder(x)
        return (
            self.head(x)
            if self.system == "flat"
            else {f"task{i + 1}": h(x) for i, h in enumerate(self.heads)}
        )


def loaders(config, *args, **kwargs):
    datasets = {"train": TinyDataset()}
    if config["system_type"] == "flat":
        datasets["validation"] = TinyDataset(split="validation")
    else:
        datasets.update(
            task1=TinyDataset(split="validation"),
            task2=TinyDataset(split="validation"),
            task3=TinyDataset(source="stage3", split="validation"),
        )
        datasets["train"] = torch.utils.data.ConcatDataset(
            [TinyDataset(), TinyDataset(source="stage3")]
        )
    return {
        k: DataLoader(
            d,
            batch_size=4,
            shuffle=k == "train",
            generator=torch.Generator().manual_seed(42),
        )
        for k, d in datasets.items()
    }


@pytest.fixture
def workspace(tmp_path, monkeypatch, config):
    for name in (
        "data/manifests",
        "experiments",
        "results/predictions",
        "results/historical",
    ):
        (tmp_path / name).mkdir(parents=True, exist_ok=True)
    for source in config["manifests"].values():
        shutil.copy(ROOT / source, tmp_path / source)
    shutil.copy(
        ROOT / "experiments/experiment_registry.csv",
        tmp_path / "experiments/experiment_registry.csv",
    )
    shutil.copy(
        ROOT / "results/historical/document_reported_internal_test.csv",
        tmp_path / "results/historical/document_reported_internal_test.csv",
    )
    for module in (training, artifacts, evaluation, statistics):
        monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(training, "training_loaders", loaders)
    monkeypatch.setattr(training, "build_model", TinyModel)
    monkeypatch.setattr(evaluation, "build_model", TinyModel)
    monkeypatch.setattr(evaluation, "SkinDataset", TinyDataset)
    monkeypatch.setattr(evaluation, "verify_images", lambda *a, **k: {"synthetic": True})
    monkeypatch.setattr(evaluation, "make_loader", lambda d, c, **kw: DataLoader(d, batch_size=4))
    c = copy.deepcopy(config)
    c["training"]["epochs"] = 2
    c["statistics"]["replicates"] = 10
    return tmp_path, c


def execute(c, root, run_id, resume=False):
    d = root / "experiments/runs" / run_id
    if not resume:
        d.mkdir(parents=True)
    return training._run(c, d, run_id, root, torch.device("cpu"), resume)


@pytest.mark.parametrize("system", ["flat", "shared_hard"])
def test_interruption_resume_matches_uninterrupted(workspace, monkeypatch, system):
    root, c = workspace
    c["system_type"] = system
    full = execute(c, root, "full")
    real = training.training_epoch
    calls = []

    def interrupted(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("synthetic VM interruption")
        return real(*args, **kwargs)

    monkeypatch.setattr(training, "training_epoch", interrupted)
    with pytest.raises(RuntimeError, match="synthetic VM"):
        execute(c, root, "resumed")
    monkeypatch.setattr(training, "training_epoch", real)
    resumed = execute(c, root, "resumed", True)
    assert resumed["status"] == full["status"] == "completed"
    states = []
    for name in ("full", "resumed"):
        ref = json.loads((root / f"experiments/runs/{name}/checkpoints.json").read_text())["last"]
        states.append(artifacts.load_checkpoint(ref))
    for k, v in states[0]["model_state_dict"].items():
        assert torch.equal(v, states[1]["model_state_dict"][k])
    assert states[0]["history"] == states[1]["history"]
    assert states[0]["scheduler_state_dict"] == states[1]["scheduler_state_dict"]
    with pytest.raises(ValueError, match="Completed"):
        execute(c, root, "resumed", True)


def test_training_to_frozen_evaluation_and_statistics(workspace):
    root, c = workspace
    for system in ("flat", "shared_hard"):
        c["system_type"] = system
        execute(c, root, system)
        result = evaluation.evaluate(system, data_root=root, device="cpu", workers=0)
        assert result["sample_count"] == 8
        assert result["metrics"]["endpoint"]["sample_count"] == 8
        meta, frame = statistics.load_evaluation(f"{system}_test")
        if system == "shared_hard":
            assert frame.loc[frame.true_class == 0, "stage2_truth"].isna().all()
            assert frame.oracle_gate.notna().all()
        with pytest.raises(FileExistsError):
            evaluation.evaluate(system, data_root=root, device="cpu", workers=0)
    result, directory = statistics.compare_evaluations("flat_test", "shared_hard_test", c)
    assert result["bootstrap"]["replicates"] == 10
    assert (directory / "historical_vs_reconstructed.csv").is_file()
    registry = pd.read_csv(root / "experiments/experiment_registry.csv")
    assert registry.status.tolist() == ["running", "completed", "running", "completed"]


def test_resume_rejects_scientific_drift(workspace, monkeypatch):
    root, c = workspace
    real = training.training_epoch
    count = []

    def interrupted(*args, **kwargs):
        count.append(1)
        if len(count) == 2:
            raise RuntimeError("interrupt")
        return real(*args, **kwargs)

    monkeypatch.setattr(training, "training_epoch", interrupted)
    with pytest.raises(RuntimeError):
        execute(c, root, "drift")
    c["training"]["lr"] *= 2
    with pytest.raises(ValueError, match="config or manifest"):
        execute(c, root, "drift", True)
