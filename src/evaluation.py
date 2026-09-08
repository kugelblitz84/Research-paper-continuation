"""Frozen-checkpoint evaluation; never imported by the training module."""

import json
from pathlib import Path

import pandas as pd
import torch

from src.artifacts import load_checkpoint, utc_now, write_json
from src.config import CLASSES, ROOT, TASK_CLASSES
from src.data import SkinDataset, make_loader, resolve_data_root, verify_images
from src.metrics import classification_metrics
from src.models import build_model
from src.reproducibility import environment, sha256
from src.routing import hard_route, oracle_route


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


@torch.inference_mode()
def prediction_frame(model, loader, system_type, device):
    model.eval()
    rows = []
    for batch in loader:
        out = model(batch["image"].to(device))
        truth = batch["target"]
        if system_type == "flat":
            logits = out.float().cpu()
            probabilities = logits.softmax(1)
            pred = logits.argmax(1)
        else:
            s1, s2 = out["task1"].float().cpu(), out["task2"].float().cpu()
            p1, p2 = s1.softmax(1), s2.softmax(1)
            pred, oracle = hard_route(s1, s2), oracle_route(truth, s2)
        for i, image_id in enumerate(batch["image_id"]):
            y, pr = int(truth[i]), int(pred[i])
            row = dict(
                image_id=image_id,
                true_class=y,
                true_label=CLASSES[y],
                predicted_class=pr,
                predicted_label=CLASSES[pr],
            )
            if system_type == "flat":
                for c in range(4):
                    row.update(
                        {
                            f"logit_{c}": float(logits[i, c]),
                            f"probability_{c}": float(probabilities[i, c]),
                        }
                    )
            else:
                row.update(
                    stage1_truth=int(y > 0),
                    stage1_prediction=int(s1[i].argmax()),
                    stage2_truth=y - 1 if y > 0 else None,
                    stage2_prediction=int(s2[i].argmax()),
                    predicted_hard_gate=pr,
                    oracle_gate=int(oracle[i]),
                )
                for stage, ls, ps in ((1, s1, p1), (2, s2, p2)):
                    for c in range(ls.shape[1]):
                        row.update(
                            {
                                f"stage{stage}_logit_{c}": float(ls[i, c]),
                                f"stage{stage}_probability_{c}": float(ps[i, c]),
                            }
                        )
                # All task heads are computed, including Stage2 on true NM; truth remains missing.
                row["stage2_used_by_hard_gate"] = bool(s1[i].argmax() == 1)
            rows.append(row)
    frame = pd.DataFrame(rows)
    if frame.empty or frame.image_id.duplicated().any():
        raise ValueError("Empty or duplicate prediction IDs")
    return frame


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
    metrics = {"endpoint": classification_metrics(frame.true_class, frame.predicted_class)}
    if config["system_type"] == "shared_hard":
        metrics["oracle"] = classification_metrics(frame.true_class, frame.oracle_gate)
        malignant = frame.true_class > 0
        metrics["stage1"] = classification_metrics(
            frame.stage1_truth, frame.stage1_prediction, TASK_CLASSES[0]
        )
        metrics["stage2_malignant_subset"] = classification_metrics(
            frame.loc[malignant, "stage2_truth"].astype(int),
            frame.loc[malignant, "stage2_prediction"],
            TASK_CLASSES[1],
        )
        metrics["routing"] = dict(
            malignant_false_negative=int(((frame.stage1_prediction == 0) & malignant).sum()),
            nonmalignant_false_positive=int(((frame.stage1_prediction == 1) & ~malignant).sum()),
            malignant_sensitivity=float((frame.loc[malignant, "stage1_prediction"] == 1).mean()),
            oracle_gap=metrics["oracle"]["macro_f1"] - metrics["endpoint"]["macro_f1"],
        )
    metadata = dict(
        run_id=run_id,
        architecture=config["architecture"],
        system_type=config["system_type"],
        split=split,
        sample_count=len(frame),
        prediction_path=str(output.relative_to(ROOT)),
        prediction_sha256=sha256(output),
        checkpoint=frozen["best"],
        manifest_hashes=payload["manifest_hashes"],
        config_hash=payload["config_hash"],
        environment=environment(),
        created_at=utc_now(),
        metrics=metrics,
    )
    write_json(directory / "evaluation.json", metadata)
    return metadata
