"""Frozen-checkpoint evaluation; never imported by the training module."""

import json
from pathlib import Path

import torch

from src.artifacts import load_checkpoint, utc_now, write_json
from src.config import ROOT
from src.data import SkinDataset, make_loader, resolve_data_root, verify_images
from src.models import build_model
from src.predictions import prediction_frame, prediction_metrics
from src.reproducibility import environment, sha256


def frozen_run(run_id):
    if Path(run_id).name != run_id:
        raise ValueError("Invalid run ID")
    run = ROOT / "experiments/runs" / run_id
    summary = json.loads((run / "run_summary.json").read_text())
    if summary["status"] != "completed":
        raise ValueError("Evaluation requires a completed frozen run")
    frozen = json.loads((run / "frozen_checkpoint.json").read_text())
    payload = load_checkpoint(frozen["best"])
    if (
        payload["config_hash"] != frozen["config_hash"]
        or payload["manifest_hashes"] != frozen["manifest_hashes"]
    ):
        raise ValueError("Frozen checkpoint provenance mismatch")
    for source, path in payload["config_snapshot"]["manifests"].items():
        if sha256(ROOT / path) != payload["manifest_hashes"][source]:
            raise ValueError("Manifest changed since training")
    return payload, frozen


def evaluate(run_id, *, data_root=None, split="test", device="cuda", workers=None):
    if split not in ("validation", "test"):
        raise ValueError("Evaluation split must be validation or test")
    payload, frozen = frozen_run(run_id)
    config = payload["config_snapshot"]
    root = resolve_data_root(data_root)
    print(f"Dataset root: {root}")
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    # Verify endpoint cohort only; T-category is not an integrated endpoint here.
    endpoint_cfg = dict(config, system_type="flat")
    verify_images(endpoint_cfg, root, splits=(split,), hashes=True)
    dataset = SkinDataset(config, "isic", split, root, verify=True)
    loader = make_loader(dataset, config, workers=workers)
    model = build_model(config, pretrained=False).to(device)
    model.load_state_dict(payload["model_state_dict"])
    frame = prediction_frame(model, loader, config["system_type"], device)
    if (
        frame.image_id.tolist() != dataset.frame.image_id.tolist()
        or frame.true_class.tolist() != dataset.frame.target.tolist()
    ):
        raise ValueError("Prediction cohort/order/truth mismatch")
    evaluation_id = f"{run_id}_{split}"
    directory = ROOT / "experiments/evaluations" / evaluation_id
    directory.mkdir(parents=True, exist_ok=False)
    output = ROOT / "results/predictions" / f"{evaluation_id}.csv"
    with output.open("x", encoding="utf-8", newline="") as handle:
        frame.to_csv(handle, index=False, float_format="%.17g")
    metrics = prediction_metrics(frame, config["system_type"])
    metadata = dict(
        run_id=run_id,
        architecture=config["architecture"],
        system_type=config["system_type"],
        split=split,
        sample_count=len(frame),
        prediction_path=str(output.relative_to(ROOT)),
        prediction_sha256=sha256(output),
        checkpoint=frozen["best"],
        components=frozen.get("components", {}),
        seed=config["seed"],
        manifest_hashes=payload["manifest_hashes"],
        config_hash=payload["config_hash"],
        environment=environment(),
        created_at=utc_now(),
        metrics=metrics,
    )
    write_json(directory / "evaluation.json", metadata)
    from src.results_registry import rebuild

    rebuild(ROOT)
    return metadata
