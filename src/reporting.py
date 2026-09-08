"""Notebook displays and saved figures from explicit disk artifacts."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.artifacts import utc_now, write_json
from src.config import ROOT, digest
from src.reproducibility import sha256


def run_status(run_id):
    d = ROOT / "experiments/runs" / run_id
    if not (d / "run_summary.json").exists():
        return {"status": "not_started", "run_id": run_id}, pd.DataFrame()
    history = pd.read_csv(d / "history.csv") if (d / "history.csv").exists() else pd.DataFrame()
    return json.loads((d / "run_summary.json").read_text()), history


def plot_history(history):
    if history.empty:
        return None
    fig, ax = plt.subplots(1, 2, figsize=(10, 3))
    history.plot(x="epoch", y="train_loss", ax=ax[0], legend=False, title="Training loss")
    history.plot(
        x="epoch",
        y="selection_score",
        ax=ax[1],
        legend=False,
        title="Validation selection score",
    )
    fig.tight_layout()
    return fig


def plot_comparison(result, output):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    systems = ("flat", "shared_hard")
    axes[0].bar(
        ["Flat", "Shared-Hard"],
        [result[s]["macro_f1"] for s in systems],
        color=["#376b91", "#c37636"],
    )
    axes[0].set(ylabel="Macro-F1", ylim=(0, 1), title="Frozen endpoint evaluation")
    classes = result["flat"]["class_names"]
    x = list(range(len(classes)))
    for i, s in enumerate(systems):
        axes[1].bar(
            [v + (i - 0.5) * 0.36 for v in x],
            [result[s]["per_class"][c]["f1"] for c in classes],
            width=0.36,
            label=s,
        )
    axes[1].set(
        xticks=x,
        xticklabels=classes,
        ylim=(0, 1),
        ylabel="F1",
        title="Classwise performance",
    )
    axes[1].tick_params(axis="x", labelrotation=20)
    axes[1].legend()
    fig.tight_layout()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as f:
        fig.savefig(f, format="png", dpi=160)
    return fig


def record_reproduction_review(config, *, reviewer, rationale, accept=False):
    """Explicit researcher review after B0 results, never an automatic score threshold."""
    if not accept or not reviewer.strip() or not rationale.strip():
        raise ValueError("Explicit acceptance, reviewer and rationale required")
    files = []
    for system in ("flat", "shared_hard"):
        path = (
            ROOT / f"experiments/evaluations/{system}_efficientnet_b0_seed42_test/evaluation.json"
        )
        meta = json.loads(path.read_text())
        if (
            meta["architecture"] != "efficientnet_b0"
            or meta["system_type"] != system
            or meta["split"] != "test"
        ):
            raise ValueError("Wrong B0 evidence")
        for p in (path, ROOT / meta["prediction_path"]):
            files.append({"path": str(p.relative_to(ROOT)), "sha256": sha256(p)})
        if sha256(ROOT / meta["prediction_path"]) != meta["prediction_sha256"]:
            raise ValueError("Prediction hash changed")
    comparison = ROOT / "results/tables/efficientnet_b0_test_comparison/statistics.json"
    files.append({"path": str(comparison.relative_to(ROOT)), "sha256": sha256(comparison)})
    protocol = {
        k: v
        for k, v in config.items()
        if k
        not in (
            "architecture",
            "system_type",
            "experiment_id",
            "config_path",
            "protocol",
        )
    }
    gate = ROOT / "experiments/reproduction_gate.json"
    if gate.exists():
        raise FileExistsError("Reproduction review already exists")
    write_json(
        gate,
        dict(
            status="accepted",
            reviewer=reviewer,
            rationale=rationale,
            reviewed_at=utc_now(),
            protocol_digest=digest(protocol),
            evidence=files,
        ),
    )
    return gate
