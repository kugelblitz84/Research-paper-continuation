"""CPU-only contracts for Block-A identities, routing, indexing, and selection."""

import copy
import json

import pandas as pd
import pytest
import torch
from torch import nn

from src.block_a import aggregate
from src.config import ARCHITECTURES, ROOT, digest
from src.experiment_catalog import SYSTEMS, expected_cells, experiment_config
from src.models import systems
from src.models.backbones import EncoderParts
from src.results_registry import (
    atomic_csv,
    collect,
    matrix_status,
    read_csv,
    rebuild,
)
from src.routing import endpoint_output, hard_route, soft_probabilities
from src.run_experiment import disposition


@pytest.mark.parametrize("system", SYSTEMS)
@pytest.mark.parametrize("backbone", ARCHITECTURES)
def test_config_identity(system, backbone):
    c = experiment_config(system, backbone)
    assert c["experiment_id"] == f"{system}_{backbone}_seed42"
    assert digest(c) == digest(experiment_config(system, backbone))
    assert digest(c) == digest(dict(reversed(list(c.items()))))
    with pytest.raises(ValueError, match="Seed"):
        experiment_config(system, backbone, 43)


@pytest.fixture
def tiny_encoder(monkeypatch):
    def build(*args, **kwargs):
        return EncoderParts(
            nn.Conv2d(3, 4, 1), nn.AdaptiveAvgPool2d(1), 4, nn.Identity(), "synthetic"
        )

    monkeypatch.setattr(systems, "build_encoder", build)


@pytest.mark.parametrize("system", ["shared_soft", "dedicated_hard", "dedicated_soft"])
def test_new_models_and_parameter_independence(tiny_encoder, system):
    model = systems.build_model(experiment_config(system, "efficientnet_b0"), pretrained=False)
    out = model(torch.zeros(2, 3, 8, 8))
    assert out["task1"].shape == (2, 2) and out["task2"].shape == (2, 3)
    assert "task3" not in out
    if system.startswith("dedicated"):
        p1, p2 = list(model.task1.parameters()), list(model.task2.parameters())
        assert not {id(p) for p in p1} & {id(p) for p in p2}
        before = [p.clone() for p in p2]
        torch.optim.SGD(p1, lr=0.1).zero_grad()
        out["task1"].sum().backward()
        torch.optim.SGD(p1, lr=0.1).step()
        assert all(torch.equal(a, b) for a, b in zip(before, p2))


@pytest.mark.parametrize("system", ["shared_soft", "dedicated_soft"])
def test_soft_endpoint(system):
    s1 = torch.tensor([[0.4, 0.6], [0.1, 0.9]]).log()
    s2 = torch.tensor([[0.34, 0.33, 0.33], [0.1, 0.2, 0.7]]).log()
    expected = torch.tensor([[0.4, 0.204, 0.198, 0.198], [0.1, 0.09, 0.18, 0.63]])
    pred, probabilities = endpoint_output({"task1": s1, "task2": s2}, system)
    assert torch.allclose(probabilities, expected)
    assert torch.allclose(probabilities.sum(1), torch.ones(2))
    assert pred.tolist() == [0, 3]
    assert hard_route(s1, s2).tolist() == [1, 3]
    with pytest.raises(ValueError):
        soft_probabilities(s1, s1)


def test_dedicated_hard():
    s1, s2 = (
        torch.tensor([[3.0, 1.0], [1.0, 3.0]]),
        torch.tensor([[0.0, 3.0, 1.0], [0.0, 1.0, 3.0]]),
    )
    pred, prob = endpoint_output({"task1": s1, "task2": s2}, "dedicated_hard")
    assert pred.tolist() == [0, 3]
    assert torch.allclose(prob.sum(1), torch.ones(2))
    assert prob.argmax(1).tolist() == pred.tolist()


def completed_fixture(root, *, system="shared_soft", score=0.75, suffix=""):
    c = experiment_config(system, "efficientnet_b0")
    run_id = c["experiment_id"]
    ref = {"path": f"models/checkpoints/{run_id}/best.pt", "sha256": "synthetic-checkpoint"}
    directory = root / "experiments/runs" / run_id
    directory.mkdir(parents=True, exist_ok=True)
    import yaml

    (directory / "config_snapshot.yaml").write_text(yaml.safe_dump(c), encoding="utf-8")
    summary = dict(
        experiment_id=run_id,
        architecture=c["architecture"],
        system_type=system,
        seed=42,
        best_epoch=3,
        selection_metric="validation_macro_f1",
        best_validation_score=score,
        status="completed",
        checkpoint_path=ref["path"],
    )
    (directory / "run_summary.json").write_text(json.dumps(summary))
    (directory / "frozen_checkpoint.json").write_text(
        json.dumps(dict(best=ref, config_hash=digest(c), manifest_hashes={}))
    )
    evaluation = root / "experiments/evaluations" / (run_id + suffix)
    evaluation.mkdir(parents=True, exist_ok=True)
    meta = dict(
        run_id=run_id,
        architecture=c["architecture"],
        system_type=system,
        seed=42,
        split="validation",
        checkpoint=ref,
        config_hash=digest(c),
        prediction_sha256="synthetic-prediction" + suffix,
        metrics={"endpoint": {"macro_f1": score, "sample_count": 8}},
    )
    (evaluation / "evaluation.json").write_text(json.dumps(meta))
    return c, evaluation, meta


def test_empty_matrix_and_rebuild(tmp_path):
    status = rebuild(tmp_path)
    assert len(status) == 35 and set(status.status) == {"PENDING"}
    assert status.validation_macro_f1.eq("").all()
    table = pd.read_csv(tmp_path / "results/block_a/validation_matrix.csv")
    assert table.iloc[:, 1:].isna().all().all()
    assert len(read_csv(tmp_path / "results/master_results.csv")) == 0
    assert (
        json.loads((tmp_path / "results/block_a/selection.json").read_text())["status"]
        == "INCOMPLETE"
    )


def test_rebuild_idempotent_and_distinct_scientific_identity(tmp_path):
    c, directory, meta = completed_fixture(tmp_path)
    first = rebuild(tmp_path)
    path = tmp_path / "results/master_results.csv"
    before = path.read_bytes()
    rebuild(tmp_path)
    assert path.read_bytes() == before
    assert len(read_csv(path)) == 1
    assert read_csv(tmp_path / "experiments/checkpoint_registry.csv")[0]["checkpoint_sha256"]
    assert first.loc[first.run_id == c["experiment_id"], "validation_macro_f1"].iloc[0] == 0.75
    completed_fixture(tmp_path, suffix="_other", score=0.76)
    status = rebuild(tmp_path)
    assert len(read_csv(path)) == 2
    assert status.loc[status.run_id == c["experiment_id"], "validation_macro_f1"].iloc[0] == ""
    # A VM-only artifact disappearing never deletes prior indexed evidence.
    (directory / "evaluation.json").unlink()
    rebuild(tmp_path)
    assert len(read_csv(path)) == 2


def test_identical_identity_conflicting_metrics_rejected(tmp_path):
    _, directory, meta = completed_fixture(tmp_path)
    rebuild(tmp_path)
    meta["metrics"]["endpoint"]["macro_f1"] = 0.1
    (directory / "evaluation.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="Conflicting canonical"):
        rebuild(tmp_path)


def test_test_results_cannot_fill_validation(tmp_path):
    c, directory, meta = completed_fixture(tmp_path)
    meta["split"] = "test"
    (directory / "evaluation.json").write_text(json.dumps(meta))
    status = rebuild(tmp_path)
    assert status.loc[status.run_id == c["experiment_id"], "validation_macro_f1"].iloc[0] == ""
    assert (
        json.loads((tmp_path / "results/block_a/selection.json").read_text())["status"]
        == "INCOMPLETE"
    )


def test_completed_protection_and_config_drift(tmp_path):
    c, _, _ = completed_fixture(tmp_path)
    assert disposition(c, root=tmp_path) == "SKIP"
    with pytest.raises(ValueError, match="Completed"):
        disposition(c, root=tmp_path, resume=True)
    changed = copy.deepcopy(c)
    changed["training"]["lr"] *= 2
    with pytest.raises(ValueError, match="Incompatible config"):
        disposition(changed, root=tmp_path)


def test_registry_only_completion_is_never_retrained(tmp_path):
    c = experiment_config("flat", "efficientnet_b0")
    path = tmp_path / "experiments/experiment_registry.csv"
    row = dict(
        experiment_id=c["experiment_id"],
        system_type="flat",
        architecture="efficientnet_b0",
        seed=42,
        status="completed",
        checkpoint_path="missing/best.pt",
        best_epoch=5,
        selection_metric="validation_macro_f1",
        best_validation_score=0.6504898607805603,
    )
    atomic_csv(path, [row], list(row))
    assert disposition(c, root=tmp_path) == "SKIP_MISSING_PROVENANCE"
    assert matrix_status(tmp_path).status.tolist().count("COMPLETED") == 1
    rows, checkpoints, _ = collect(tmp_path)
    assert rows[0]["macro_f1"] == "0.6504898607805603"
    assert checkpoints[0]["checkpoint_sha256"] == ""


def test_h1_mean_is_not_endpoint(tmp_path):
    c = experiment_config("shared_hard", "efficientnet_b0")
    row = dict(
        experiment_id=c["experiment_id"],
        system_type="shared_hard",
        architecture="efficientnet_b0",
        seed=42,
        status="completed",
        selection_metric="mean_task1_task2_task3_macro_f1",
        best_validation_score=0.8,
    )
    atomic_csv(tmp_path / "experiments/experiment_registry.csv", [row], list(row))
    rows, _, _ = collect(tmp_path)
    assert not rows


def filled_status(scores):
    rows = expected_cells()
    for row in rows:
        row.update(status="COMPLETED", validation_macro_f1=scores[row["system_type"]])
    return pd.DataFrame(rows)


@pytest.mark.parametrize("gap,tie", [(0.004, True), (0.005, True), (0.006, False)])
def test_aggregation_means_deltas_and_near_tie(gap, tie):
    scores = dict(
        flat=0.65, shared_hard=0.70, shared_soft=0.8, dedicated_hard=0.8 - gap, dedicated_soft=0.72
    )
    matrix, summary, decision = aggregate(filled_status(scores))
    assert matrix.Backbone.tolist()[-2:] == ["Mean", "Median"]
    assert matrix.Flat.iloc[-1] == pytest.approx(0.65)
    row = summary.set_index("system_type").loc["shared_soft"]
    assert row["mean"] == pytest.approx(0.8)
    assert row["median"] == pytest.approx(0.8)
    assert row.delta_vs_flat == pytest.approx(0.15)
    assert row.delta_vs_h1 == pytest.approx(0.10)
    assert decision["status"] == ("NEAR_TIE" if tie else "SELECTED")
    assert len(decision["advancing"]) == (2 if tie else 1)


def test_missing_is_not_zero_or_selectable():
    status = filled_status(dict.fromkeys(SYSTEMS, 0.7))
    status.loc[0, "validation_macro_f1"] = float("nan")
    matrix, summary, decision = aggregate(status)
    assert pd.isna(matrix.Flat.iloc[0])
    assert matrix.Flat.iloc[-2] == pytest.approx(0.7)
    assert decision["status"] == "INCOMPLETE"
    assert summary.iloc[0]["count"] == 6


def test_training_and_runner_never_import_evaluation():
    import ast

    for filename in ["training.py", "validation_results.py", "run_experiment.py", "run_block_a.py"]:
        tree = ast.parse((ROOT / "src" / filename).read_text(encoding="utf-8-sig"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert node.module != "src.evaluation"
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                assert node.func.id != "evaluate"


def test_new_training_cannot_guess_undefined_selection():
    from src.training import train

    c = experiment_config("shared_soft", "efficientnet_b0")
    if "shared_soft" not in c["selection"]:
        with pytest.raises(ValueError, match="explicitly approved"):
            train(c, allow_training=True)
