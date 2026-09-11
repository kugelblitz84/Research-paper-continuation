"""Read-only scientific backfill with atomic, lossless derived CSV indexes.

No model loading, data loading, inference or training occurs here. Previously indexed
rows survive missing VM artifacts. Changed scientific identities remain distinct.
"""

import argparse
import csv
import json
import os
import uuid
from pathlib import Path

import pandas as pd
import yaml

from src.artifacts import exclusive_lock
from src.config import CLASSES, ROOT, digest
from src.experiment_catalog import SYSTEMS, expected_cells, experiment_config
from src.reproducibility import sha256

METRICS = [
    "sample_count",
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "weighted_f1",
    "macro_precision",
    "macro_recall",
]
IDENTITY = [
    "run_id",
    "system_code",
    "system_type",
    "backbone",
    "seed",
    "split",
    "evaluation_mode",
    "status",
    "best_epoch",
    "selection_metric",
    "selection_score",
]
PROVENANCE = [
    "checkpoint_path",
    "checkpoint_sha256",
    "task1_checkpoint_path",
    "task1_checkpoint_sha256",
    "task2_checkpoint_path",
    "task2_checkpoint_sha256",
    "prediction_path",
    "prediction_sha256",
    "config_hash",
    "manifest_hash_isic",
    "manifest_hash_stage3",
    "protocol_version",
    "protocol_hash",
    "extension_hash",
    "git_commit",
    "python_version",
    "torch_version",
    "cuda_version",
    "gpu",
    "training_duration_seconds",
    "gpu_hours",
    "parameter_count",
    "model_count",
    "source_path",
    "source_sha256",
    "created_at",
    "confusion_matrix",
    "metrics_json",
]
RESULT_COLUMNS = (
    ["result_id"]
    + IDENTITY
    + METRICS
    + [f"{c}_{m}" for c in CLASSES for m in ("precision", "recall", "f1", "support")]
    + [
        "malignant_false_negative",
        "nonmalignant_false_positive",
        "malignant_sensitivity",
        "oracle_gap",
    ]
    + PROVENANCE
)
CHECKPOINT_COLUMNS = [
    "checkpoint_id",
    "run_id",
    "system_type",
    "backbone",
    "seed",
    "component",
    "best_epoch",
    "selection_metric",
    "selection_score",
    "checkpoint_path",
    "checkpoint_sha256",
    "config_hash",
    "status",
    "source_path",
    "state_dict_key",
]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig")) if path.is_file() else {}


def read_csv(path):
    if not path.is_file():
        return []
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def atomic_csv(path, rows, columns):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with temp.open("x", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="raise", lineterminator="\n")
        writer.writeheader()
        writer.writerows({c: r.get(c, "") for c in columns} for r in rows)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def _identity(row):
    keys = [
        "run_id",
        "split",
        "evaluation_mode",
        "config_hash",
        "checkpoint_path",
        "checkpoint_sha256",
        "task1_checkpoint_sha256",
        "task2_checkpoint_sha256",
        "prediction_sha256",
        "manifest_hash_isic",
        "manifest_hash_stage3",
    ]
    values = {k: row.get(k) or "" for k in keys}
    if not values["prediction_sha256"]:
        values["source_sha256"] = row.get("source_sha256", "")
    return digest(values)


def _merge(old, new, key, *, scientific=False):
    merged = {r[key]: r for r in old}
    for row in new:
        prior = merged.get(row[key])
        if prior and scientific:
            # Same immutable inputs cannot legitimately yield different saved metrics.
            for field in METRICS + ["metrics_json"]:
                a, b = prior.get(field), row.get(field)
                if a not in (None, "") and b not in (None, ""):
                    same = str(a) == str(b)
                    if field != "metrics_json":
                        same = float(a) == float(b)
                    if not same:
                        raise ValueError(f"Conflicting canonical result {row[key]}: {field}")
        merged[row[key]] = {**(prior or {}), **row}
    return sorted(merged.values(), key=lambda r: r[key])


def run_evidence(root=ROOT):
    """Keep the event log intact; latest event plus canonical summary describe state."""
    found = {}
    for row in read_csv(root / "experiments/experiment_registry.csv"):
        run_id = row["experiment_id"]
        current = found.setdefault(run_id, {"events": []})
        current["events"].append(row)
        current["summary"] = row
    for directory in sorted((root / "experiments/runs").glob("*")):
        if not directory.is_dir():
            continue
        current = found.setdefault(directory.name, {"events": []})
        summary = read_json(directory / "run_summary.json")
        if summary:
            current["summary"] = summary
        else:
            current.setdefault("summary", {"status": "incomplete"})
    for run_id, item in found.items():
        directory = root / "experiments/runs" / run_id
        item["frozen"] = read_json(directory / "frozen_checkpoint.json")
        item["environment"] = read_json(directory / "environment.json")
        path = directory / "config_snapshot.yaml"
        item["config"] = yaml.safe_load(path.read_text(encoding="utf-8")) if path.is_file() else {}
        if (
            item["config"]
            and item["frozen"]
            and digest(item["config"]) != item["frozen"].get("config_hash")
        ):
            raise ValueError(f"Config snapshot/frozen provenance mismatch: {run_id}")
        item["directory_present"] = directory.is_dir()
        item["active"] = (directory / "active.lock").is_file()
    return found


def _base(run_id, evidence):
    summary, config = evidence.get("summary", {}), evidence.get("config", {})
    frozen = evidence.get("frozen", {})
    system = summary.get("system_type", config.get("system_type", ""))
    ref = frozen.get("best", {})
    return dict(
        run_id=run_id,
        system_type=system,
        system_code=SYSTEMS.get(system, ("",))[0],
        backbone=summary.get("architecture", config.get("architecture", "")),
        seed=summary.get("seed", config.get("seed", "")),
        best_epoch=summary.get("best_epoch", ""),
        selection_metric=summary.get("selection_metric", ""),
        selection_score=summary.get("best_validation_score", ""),
        checkpoint_path=ref.get("path", summary.get("checkpoint_path", "")),
        checkpoint_sha256=ref.get("sha256", ""),
        config_hash=frozen.get("config_hash", digest(config) if config else ""),
        protocol_version=config.get("protocol_version", ""),
        extension_hash=digest(config["block_a_extension"]) if "block_a_extension" in config else "",
    )


def _result_rows(meta, path, root, evidence):
    run_id = meta["run_id"]
    item = evidence.get(run_id, {})
    base = _base(run_id, item)
    evaluated_path = meta.get("checkpoint", {}).get("path")
    if evaluated_path and evaluated_path != base.get("checkpoint_path"):
        # An older evaluation must not inherit the latest checkpoint's epoch/score.
        matching = [
            event
            for event in item.get("events", [])
            if event.get("checkpoint_path") == evaluated_path
        ]
        old = matching[-1] if matching else {}
        base.update(
            best_epoch=old.get("best_epoch", ""),
            selection_score=old.get("best_validation_score", ""),
            selection_metric=old.get("selection_metric", ""),
            config_hash="",
        )
    base.update({k: meta[k] for k in base if k in meta})
    base.update(
        backbone=meta.get("architecture", base["backbone"]),
        split=meta["split"],
        status="COMPLETED",
        prediction_path=meta.get("prediction_path", ""),
        prediction_sha256=meta.get("prediction_sha256", ""),
        source_path=path.relative_to(root).as_posix(),
        source_sha256=sha256(path),
        created_at=meta.get("created_at", ""),
    )
    checkpoint = meta.get("checkpoint", {})
    for k in ("path", "sha256"):
        base["checkpoint_" + k] = checkpoint.get(k, base.get("checkpoint_" + k, ""))
    for component, ref in meta.get("components", {}).items():
        for k in ("path", "sha256"):
            base[f"{component}_checkpoint_{k}"] = ref.get(k, "")
    for source, value in meta.get("manifest_hashes", {}).items():
        base["manifest_hash_" + source] = value
    env = meta.get("environment", {})
    for source, target in [
        ("python", "python_version"),
        ("torch", "torch_version"),
        ("cuda", "cuda_version"),
        ("gpu", "gpu"),
        ("git_commit", "git_commit"),
    ]:
        base[target] = env.get(source, "")
    for k in (
        "protocol_hash",
        "training_duration_seconds",
        "gpu_hours",
        "parameter_count",
        "model_count",
    ):
        base[k] = meta.get(k, "")
    prediction = root / base["prediction_path"] if base["prediction_path"] else None
    if prediction and prediction.is_file() and base["prediction_sha256"]:
        if sha256(prediction) != base["prediction_sha256"]:
            raise ValueError(f"Prediction hash mismatch: {prediction}")
    rows = []
    for mode, metrics in meta["metrics"].items():
        if mode == "routing":
            continue
        row = dict(base, evaluation_mode=mode)
        row.update({k: metrics[k] for k in METRICS if k in metrics})
        row.update(meta["metrics"].get("routing", {}) if mode == "endpoint" else {})
        for cls, values in metrics.get("per_class", {}).items():
            if cls in CLASSES:
                row.update({f"{cls}_{k}": v for k, v in values.items()})
        row["confusion_matrix"] = json.dumps(metrics.get("confusion_matrix", []))
        row["metrics_json"] = json.dumps(metrics, sort_keys=True, separators=(",", ":"))
        row["result_id"] = _identity(row)
        rows.append(row)
    return rows


def collect(root=ROOT):
    root = Path(root)
    evidence = run_evidence(root)
    rows, checkpoints = [], []
    for path in sorted((root / "experiments/evaluations").rglob("evaluation.json")):
        rows.extend(_result_rows(read_json(path), path, root, evidence))
    for run_id, item in evidence.items():
        base = _base(run_id, item)
        # Every historical completed checkpoint event remains in the index.
        events = [e for e in item["events"] if e.get("checkpoint_path")]
        for event in events:
            checkpoint = {
                **base,
                "checkpoint_path": event["checkpoint_path"],
                "checkpoint_sha256": "",
                "config_hash": "",
                "best_epoch": event["best_epoch"],
                "selection_score": event["best_validation_score"],
            }
            checkpoint.update(
                component="endpoint" if base["system_type"] == "flat" else "shared",
                status=event["status"].upper(),
                source_path="experiments/experiment_registry.csv",
            )
            checkpoints.append(checkpoint)
        frozen = item["frozen"]
        if frozen:
            refs = {"endpoint" if base["system_type"] == "flat" else "shared": frozen["best"]}
            if base["system_type"].startswith("dedicated"):
                refs = {"system": frozen["best"], **frozen.get("components", {})}
            for component, ref in refs.items():
                checkpoints.append(
                    dict(
                        base,
                        component=component,
                        checkpoint_path=ref["path"],
                        checkpoint_sha256=ref.get("sha256", ""),
                        state_dict_key=ref.get("state_dict_key", ""),
                        status=item["summary"]["status"].upper(),
                        source_path=f"experiments/runs/{run_id}/frozen_checkpoint.json",
                    )
                )
        # Registry Flat selection is endpoint F1. Never reinterpret H1's three-task mean.
        if not any(
            r["run_id"] == run_id
            and r["split"] == "validation"
            and r["evaluation_mode"] == "endpoint"
            for r in rows
        ):
            summary = item["summary"]
            if (
                summary.get("status") == "completed"
                and summary.get("selection_metric") == "validation_macro_f1"
                and summary.get("best_validation_score") not in (None, "")
            ):
                row = dict(
                    base,
                    split="validation",
                    evaluation_mode="endpoint",
                    status="REGISTRY_ONLY",
                    macro_f1=summary["best_validation_score"],
                    source_path="experiments/experiment_registry.csv",
                    source_sha256=digest(summary),
                )
                row["result_id"] = _identity(row)
                rows.append(row)
    for row in checkpoints:
        row["checkpoint_id"] = digest(
            {
                k: row.get(k, "")
                for k in ("run_id", "component", "checkpoint_path", "checkpoint_sha256")
            }
        )
    return rows, checkpoints, evidence


def matrix_status(root=ROOT, seed=42, *, evidence=None, results=None):
    root = Path(root)
    if evidence is None:
        results, _, evidence = collect(root)
    results = results or []
    results = _merge(
        read_csv(root / "results/master_results.csv"), results, "result_id", scientific=True
    )
    rows = []
    for cell in expected_cells(seed):
        item = evidence.get(cell["run_id"], {})
        base = _base(cell["run_id"], item)
        summary = item.get("summary", {})
        raw = summary.get("status", "pending").lower()
        state = {
            "completed": "COMPLETED",
            "pending": "PENDING",
            "running": "INCOMPLETE",
            "failed": "FAILED",
            "interrupted": "FAILED",
        }.get(raw, "INCOMPLETE")
        if raw == "running" and item.get("active"):
            state = "RUNNING"
        config = experiment_config(cell["system_type"], cell["backbone"], seed)
        expected_hash = digest(config)
        actual_hash = base.get("config_hash", "")
        note = ""
        if actual_hash and actual_hash != expected_hash:
            state, note = "INCOMPLETE", "INCOMPATIBLE_CONFIG"
        if raw == "completed" and not item.get("frozen"):
            note = "Registry completion; canonical VM artifacts unavailable locally"
        candidates = [
            r
            for r in results
            if r["run_id"] == cell["run_id"]
            and r["split"] == "validation"
            and r["evaluation_mode"] == "endpoint"
            and r.get("config_hash") == expected_hash
            and r.get("checkpoint_path") == base["checkpoint_path"]
            and r.get("status") == "COMPLETED"
        ]
        ids = {r["result_id"] for r in candidates}
        endpoint = candidates[0].get("macro_f1", "") if len(ids) == 1 else ""
        if len(ids) > 1:
            note = "Ambiguous validation evidence; explicit result selection required"
        rows.append(
            {
                **base,
                **cell,
                "status": state,
                "expected_config_hash": expected_hash,
                "validation_macro_f1": endpoint,
                "checkpoint_available": bool(
                    base["checkpoint_path"] and (root / base["checkpoint_path"]).is_file()
                ),
                "validation_result_id": candidates[0]["result_id"] if len(ids) == 1 else "",
                "note": note,
            }
        )
    return pd.DataFrame(rows)


def rebuild(root=ROOT, seed=42):
    root = Path(root)
    with exclusive_lock(root / "experiments/results_index.lock"):
        new, checkpoints, evidence = collect(root)
        result_path = root / "results/master_results.csv"
        checkpoint_path = root / "experiments/checkpoint_registry.csv"
        rows = _merge(read_csv(result_path), new, "result_id", scientific=True)
        refs = _merge(read_csv(checkpoint_path), checkpoints, "checkpoint_id")
        status = matrix_status(root, seed, evidence=evidence, results=rows)
        from src.block_a import aggregate

        matrix, summary, selection = aggregate(status)
        atomic_csv(result_path, rows, RESULT_COLUMNS)
        atomic_csv(checkpoint_path, refs, CHECKPOINT_COLUMNS)
        destination = root / "results/block_a"
        for name, frame in [
            ("status", status),
            ("validation_matrix", matrix),
            ("system_summary", summary),
        ]:
            atomic_csv(
                destination / f"{name}.csv",
                frame.astype(object).where(frame.notna(), "").to_dict("records"),
                frame.columns.tolist(),
            )
        from src.artifacts import write_json

        write_json(destination / "selection.json", selection)
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["rebuild"])
    args = parser.parse_args()
    if args.command == "rebuild":
        status = rebuild()
        print(status[["system_code", "backbone", "status", "note"]].to_string(index=False))
        print("Missing VM artifacts remain missing. No training or inference was performed.")


if __name__ == "__main__":
    main()
