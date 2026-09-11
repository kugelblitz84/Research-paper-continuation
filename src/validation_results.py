"""Persist endpoint validation evidence at the already-selected checkpoint.

This module accepts the training validation loader. It cannot construct test data.
"""

from src.artifacts import load_checkpoint, utc_now, write_json
from src.config import ROOT, digest
from src.predictions import prediction_frame, prediction_metrics
from src.reproducibility import environment, sha256


def persist_validation(
    model, loader, config, device, run_id, frozen, *, root=ROOT, duration_seconds=None
):
    payload = load_checkpoint(frozen["best"])
    if payload["config_hash"] != digest(config) or payload["run_id"] != run_id:
        raise ValueError("Selected checkpoint identity mismatch")
    model.load_state_dict(payload["model_state_dict"])
    frame = prediction_frame(model, loader, config["system_type"], device)
    dataset = loader.dataset
    if hasattr(dataset, "frame"):
        if (
            frame.image_id.tolist() != dataset.frame.image_id.tolist()
            or frame.true_class.tolist() != dataset.frame.target.tolist()
        ):
            raise ValueError("Validation cohort/order/truth mismatch")
    # Stable content identity makes finalization retryable after an interruption.
    evaluation_id = f"{run_id}_validation_{frozen['best']['sha256'][:16]}"
    directory = root / "experiments/evaluations" / evaluation_id
    output = root / "results/predictions" / f"{evaluation_id}.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    contents = frame.to_csv(index=False, float_format="%.17g", lineterminator="\n").encode("utf-8")
    if output.exists():
        if output.read_bytes() != contents:
            raise ValueError("Existing validation predictions differ; refusing overwrite")
    else:
        with output.open("xb") as f:
            f.write(contents)
    env = environment()
    now = utc_now()
    duration = duration_seconds
    metadata = dict(
        run_id=run_id,
        architecture=config["architecture"],
        system_type=config["system_type"],
        seed=config["seed"],
        split="validation",
        sample_count=len(frame),
        prediction_path=output.relative_to(root).as_posix(),
        prediction_sha256=sha256(output),
        checkpoint=frozen["best"],
        components=frozen.get("components", {}),
        config_hash=digest(config),
        manifest_hashes=frozen["manifest_hashes"],
        protocol_version=config["protocol_version"],
        protocol_hash=payload.get("protocol_hash", ""),
        environment=env,
        created_at=now,
        metrics=prediction_metrics(frame, config["system_type"]),
        best_epoch=payload["best_epoch"],
        selection_metric=payload.get(
            "selection_metric", config["selection"].get(config["system_type"], "")
        ),
        selection_score=payload["best_validation_score"],
        training_duration_seconds=duration,
        gpu_hours=duration / 3600 if duration is not None and device.type == "cuda" else None,
        parameter_count=sum(p.numel() for p in model.parameters()),
        model_count=2 if config["system_type"].startswith("dedicated") else 1,
    )
    path = directory / "evaluation.json"
    if path.exists():
        from src.results_registry import read_json

        prior = read_json(path)
        for key in ("prediction_sha256", "config_hash", "checkpoint", "metrics"):
            if prior[key] != metadata[key]:
                raise ValueError("Conflicting validation metadata; refusing overwrite")
        return prior
    write_json(path, metadata)
    return metadata
