"""One resumable orchestration loop; new scientific policies require explicit configuration."""

import json
import time
import traceback
import uuid
from pathlib import Path

import pandas as pd
import torch
import yaml

from src.artifacts import (
    append_registry,
    exclusive_lock,
    load_checkpoint,
    save_checkpoint,
    utc_now,
    write_json,
)
from src.config import CLASSES, ROOT, TASK_CLASSES, baseline_protocol, digest, validate_runtime_config
from src.data import audit_manifests, resolve_data_root, training_loaders, verify_images
from src.losses import MaskedLoss
from src.metrics import classification_metrics
from src.models import build_model
from src.reproducibility import (
    capture_rng,
    environment,
    restore_rng,
    seed_everything,
    sha256,
)
from src.routing import endpoint_output


def preflight(config, data_root=None, *, verify_hashes=True):
    root = resolve_data_root(data_root)
    print(f"Dataset root: {root}")
    audit = audit_manifests(config)
    if audit["cross_source_split_conflicts"]:
        raise ValueError("Cross-source split conflicts require protocol review")
    images = verify_images(config, root, hashes=verify_hashes)
    return {
        "dataset_root": str(root),
        "manifest_audit": audit,
        "images": images,
        "training_performed": False,
    }


def optimizer_components(model, config, device):
    p = config["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=p["lr"],
        weight_decay=p["weight_decay"],
        betas=tuple(p["betas"]),
        eps=p["eps"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=p["t_max"], eta_min=p["eta_min"]
    )
    scaler = torch.amp.GradScaler("cuda", enabled=p["amp"] and device.type == "cuda")
    return optimizer, scheduler, scaler


def training_epoch(model, loader, config, device, optimizer, scaler, criterion):
    model.train()
    total, n = 0.0, 0
    task_sums, task_counts = (
        {f"task{i}": 0.0 for i in (1, 2, 3)},
        {f"task{i}": 0 for i in (1, 2, 3)},
    )
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device.type,
            dtype=torch.float16,
            enabled=config["training"]["amp"] and device.type == "cuda",
        ):
            if config["system_type"].startswith("dedicated"):
                # Task-2 sees only labelled malignant training images, including BN updates.
                mask = batch["masks"][:, 1].to(device)
                task2 = x.new_zeros((len(x), 3))
                if mask.any():
                    task2 = task2.index_copy(
                        0, mask.nonzero().flatten(), model.task2(x[mask]).to(x.dtype)
                    )
                outputs = {"task1": model.task1(x), "task2": task2}
            else:
                outputs = model(x)
            if config["system_type"] == "flat":
                loss = torch.nn.functional.cross_entropy(outputs, batch["target"].to(device))
            else:
                loss, parts, counts = criterion(
                    outputs, batch["targets"].to(device), batch["masks"].to(device)
                )
                for k, value in parts.items():
                    if value is not None:
                        task_sums[k] += float(value.detach()) * counts[k]
                        task_counts[k] += counts[k]
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total += float(loss.detach()) * len(x)
        n += len(x)
    if not n:
        raise ValueError("Empty training loader")
    return {
        "train_loss": total / n,
        **{
            f"train_{k}_loss": task_sums[k] / task_counts[k] if task_counts[k] else None
            for k in task_sums
        },
    }


@torch.no_grad()
def validation_epoch(model, loaders, config, device):
    model.eval()
    scores = {}
    if config["system_type"] not in ("flat", "shared_hard"):
        truth, predicted = [], []
        for batch in loaders["task1"]:
            out = model(batch["image"].to(device))
            pred, _ = endpoint_output(out, config["system_type"])
            truth.extend(batch["target"].tolist())
            predicted.extend(pred.cpu().tolist())
        score = classification_metrics(truth, predicted, CLASSES)["macro_f1"]
        if config["selection"].get(config["system_type"]) != "validation_macro_f1":
            raise ValueError("Unsupported new-system validation selection policy")
        return {"val_validation_macro_f1": score, "selection_score": score}
    keys = ("validation",) if config["system_type"] == "flat" else ("task1", "task2", "task3")
    for key in keys:
        y, p = [], []
        for batch in loaders[key]:
            x = batch["image"].to(device)
            with torch.autocast(
                device.type,
                dtype=torch.float16,
                enabled=config["system_type"] == "flat"
                and config["training"]["amp"]
                and device.type == "cuda",
            ):
                output = model(x)
            if key == "validation":
                truth, logits, classes = batch["target"], output, CLASSES
            else:
                i = int(key[-1]) - 1
                mask = batch["masks"][:, i]
                truth, logits, classes = (
                    batch["targets"][mask, i],
                    output[key].cpu()[mask],
                    TASK_CLASSES[i],
                )
            y.extend(truth.tolist())
            p.extend(logits.argmax(1).cpu().tolist())
        scores[f"val_{key}_macro_f1"] = classification_metrics(y, p, classes)["macro_f1"]
    scores["selection_score"] = sum(scores.values()) / len(scores)
    return scores


def _allow_backbone(config):
    if config["architecture"] == "efficientnet_b0":
        return
    gate = ROOT / "experiments/reproduction_gate.json"
    if not gate.is_file():
        raise RuntimeError("Remaining six pairs are locked until B0 reproduction review")
    approval = json.loads(gate.read_text())
    if approval.get("status") != "accepted" or approval.get("protocol_digest") != digest(
        baseline_protocol(config)
    ):
        raise RuntimeError("B0 reproduction gate absent or incompatible")
    for ref in approval["evidence"]:
        if sha256(ROOT / ref["path"]) != ref["sha256"]:
            raise ValueError("Reproduction gate evidence changed")


def train(
    config,
    *,
    data_root=None,
    allow_training=False,
    resume=False,
    device="cuda",
    experiment_id=None,
):
    """Notebook entry point. Real training requires an explicit opt-in after handover."""
    if not allow_training:
        return {
            "status": "not_started",
            "reason": "Build gate: real training is disabled",
            "experiment_id": config["experiment_id"],
        }
    validate_runtime_config(config)
    from src.run_experiment import disposition

    if experiment_id is not None and experiment_id != config["experiment_id"]:
        raise ValueError("Run ID must match the resolved experiment identity")
    action = disposition(config, root=ROOT, resume=resume)
    if action not in ("RUN", "RESUME"):
        return {"experiment_id": config["experiment_id"], "status": action}
    if config["system_type"] not in config["selection"]:
        raise ValueError(
            "Unsupported checkpoint-selection policy in resolved configuration"
        )
    _allow_backbone(config)
    preflight(config, data_root)
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    run_id = experiment_id or config["experiment_id"]
    if Path(run_id).name != run_id or run_id in (".", ".."):
        raise ValueError("Invalid experiment ID")
    directory = ROOT / "experiments/runs" / run_id
    if not resume:
        directory.mkdir(parents=True, exist_ok=False)
    elif not directory.is_dir():
        raise FileNotFoundError("Resume run not found")
    with exclusive_lock(directory / "active.lock"):
        return _run(config, directory, run_id, data_root, device, resume)


def _run(config, directory, run_id, data_root, device, resume):
    started_monotonic = time.monotonic()
    seed_everything(config["seed"])
    manifest_hashes = {s: sha256(ROOT / p) for s, p in config["manifests"].items()}
    config_hash = digest(config)
    state = None
    if resume:
        summary = json.loads((directory / "run_summary.json").read_text())
        if summary["status"] == "completed":
            raise ValueError("Completed experiment cannot be resumed or overwritten")
        pointer = json.loads((directory / "checkpoints.json").read_text())
        state = load_checkpoint(pointer["last"])
        if (
            state["config_hash"] != config_hash
            or state["manifest_hashes"] != manifest_hashes
            or state["run_id"] != run_id
        ):
            raise ValueError("Resume config or manifest mismatch")
    else:
        (directory / "logs").mkdir()
        (directory / "config_snapshot.yaml").write_text(
            yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
        )
        write_json(directory / "environment.json", environment())
    row = dict(
        experiment_id=run_id,
        architecture=config["architecture"],
        system_type=config["system_type"],
        seed=config["seed"],
        config_path=config["config_path"],
        manifest_hash=digest(manifest_hashes),
        start_time=state["start_time"] if state else utc_now(),
        end_time="",
        best_epoch=0,
        selection_metric=config["selection"][config["system_type"]],
        best_validation_score="",
        checkpoint_path="",
        status="running",
        config_hash=config_hash,
        protocol_hash=digest(baseline_protocol(config)),
    )
    if "extension_hash" in config:
        row.update(extension_version=config["extension_version"], extension_hash=config["extension_hash"])
    append_registry(row)
    write_json(directory / "run_summary.json", row)
    try:
        loaders = training_loaders(config, resolve_data_root(data_root))
        model = build_model(config, pretrained=False if resume else None).to(device)
        optimizer, scheduler, scaler = optimizer_components(model, config, device)
        criterion = MaskedLoss(config).to(device) if config["system_type"] != "flat" else None
        history, best_score, best_epoch, bad, start = [], -1.0, 0, 0, 1
        best_ref = None
        if state:
            model.load_state_dict(state["model_state_dict"])
            optimizer.load_state_dict(state["optimizer_state_dict"])
            scheduler.load_state_dict(state["scheduler_state_dict"])
            scaler.load_state_dict(state["scaler_state_dict"])
            history, best_score, best_epoch, bad, start = (
                state["history"],
                state["best_validation_score"],
                state["best_epoch"],
                state["early_stopping_counter"],
                state["epoch"] + 1,
            )
            best_ref = pointer["best"]
            restore_rng(state["rng_state"], loaders)
        for epoch in range(start, config["training"]["epochs"] + 1):
            if bad >= config["training"]["patience"]:
                break
            record = {"epoch": epoch, "learning_rate": optimizer.param_groups[0]["lr"]}
            record.update(
                training_epoch(
                    model,
                    loaders["train"],
                    config,
                    device,
                    optimizer,
                    scaler,
                    criterion,
                )
            )
            record.update(validation_epoch(model, loaders, config, device))
            score = record["selection_score"]
            improved = score > best_score
            if improved:
                best_score, best_epoch, bad = score, epoch, 0
            else:
                bad += 1
            scheduler.step()
            record["patience_counter"] = bad
            history.append(record)
            payload = dict(
                model_state_dict=model.state_dict(),
                optimizer_state_dict=optimizer.state_dict(),
                scheduler_state_dict=scheduler.state_dict(),
                scaler_state_dict=scaler.state_dict(),
                epoch=epoch,
                best_validation_score=best_score,
                best_epoch=best_epoch,
                early_stopping_counter=bad,
                architecture=config["architecture"],
                pretrained_weights_resolved=getattr(model, "weight_name", "synthetic_test_model"),
                system_type=config["system_type"],
                seed=config["seed"],
                class_mappings={
                    "endpoint": dict(zip(CLASSES, range(4))),
                    **{
                        f"task{i + 1}": dict(zip(c, range(len(c))))
                        for i, c in enumerate(TASK_CLASSES)
                    },
                },
                config_snapshot=config,
                protocol_hash=row["protocol_hash"],
                selection_metric=row["selection_metric"],
                config_hash=config_hash,
                manifest_hashes=manifest_hashes,
                torch_version=str(torch.__version__),
                torchvision_version=environment()["torchvision"],
                environment=environment(),
                rng_state=capture_rng(loaders),
                history=history,
                training_duration_seconds=(
                    state.get("training_duration_seconds", 0) if state else 0
                )
                + time.monotonic()
                - started_monotonic,
                start_time=row["start_time"],
                run_id=run_id,
            )
            if "extension_hash" in config:
                payload.update(extension_version=config["extension_version"], extension_hash=config["extension_hash"])
            generation = (
                ROOT / "models/checkpoints" / run_id / f"epoch_{epoch:03d}_{uuid.uuid4().hex[:8]}"
            )
            last_ref = save_checkpoint(generation / "last.pt", payload)
            if improved:
                best_ref = save_checkpoint(generation / "best.pt", payload)
            write_json(directory / "checkpoints.json", {"last": last_ref, "best": best_ref})
            pd.DataFrame(history).to_csv(directory / "history.csv", index=False)
            row.update(
                best_epoch=best_epoch,
                best_validation_score=best_score,
                checkpoint_path=best_ref["path"],
            )
            write_json(directory / "run_summary.json", row)
            line = json.dumps(record, allow_nan=False)
            with (directory / "logs/training.jsonl").open("a", encoding="utf-8") as log:
                log.write(line + "\n")
            print(line, flush=True)
        if best_ref is None:
            raise RuntimeError("No validation-selected checkpoint")
        frozen = {
            "best": best_ref,
            "config_hash": config_hash,
            "manifest_hashes": manifest_hashes,
            "frozen_at": utc_now(),
        }
        if config["system_type"].startswith("dedicated"):
            selected = load_checkpoint(best_ref)
            components = {}
            for name in ("task1", "task2"):
                component_path = ROOT / Path(best_ref["path"]).parent / f"{name}_best.pt"
                # Separate immutable component files plus the joint resume/system checkpoint.
                component_payload = {
                    k: selected[k]
                    for k in (
                        "run_id",
                        "config_hash",
                        "manifest_hashes",
                        "best_epoch",
                        "best_validation_score",
                        "architecture",
                        "system_type",
                        "seed",
                        "config_snapshot",
                    )
                }
                component_payload.update(
                    component=name,
                    model_state_dict={
                        k[len(name) + 1 :]: v
                        for k, v in selected["model_state_dict"].items()
                        if k.startswith(name + ".")
                    },
                )
                if component_path.exists():
                    ref = {
                        "path": component_path.relative_to(ROOT).as_posix(),
                        "sha256": sha256(component_path),
                    }
                    prior = load_checkpoint(ref)
                    if (
                        prior["config_hash"] != config_hash
                        or prior["component"] != name
                        or prior["best_epoch"] != best_epoch
                        or prior["model_state_dict"].keys()
                        != component_payload["model_state_dict"].keys()
                        or any(
                            not torch.equal(v, prior["model_state_dict"][k])
                            for k, v in component_payload["model_state_dict"].items()
                        )
                    ):
                        raise ValueError("Existing component checkpoint differs")
                else:
                    ref = save_checkpoint(component_path, component_payload)
                components[name] = ref
            frozen["components"] = components
        from src.validation_results import persist_validation

        validation_loader = (
            loaders["validation"] if config["system_type"] == "flat" else loaders["task1"]
        )
        validation = persist_validation(
            model,
            validation_loader,
            config,
            device,
            run_id,
            frozen,
            root=ROOT,
            duration_seconds=(state.get("training_duration_seconds", 0) if state else 0)
            + time.monotonic()
            - started_monotonic,
        )
        write_json(directory / "frozen_checkpoint.json", frozen)
        row.update(
            status="completed",
            end_time=utc_now(),
            best_epoch=best_epoch,
            best_validation_score=best_score,
            checkpoint_path=best_ref["path"],
            checkpoint_sha256=best_ref["sha256"],
            validation_macro_f1=validation["metrics"]["endpoint"]["macro_f1"],
            training_duration_seconds=(state.get("training_duration_seconds", 0) if state else 0)
            + time.monotonic()
            - started_monotonic,
        )
    except BaseException:
        row.update(status="interrupted", end_time=utc_now())
        with (directory / "logs/error.log").open("a", encoding="utf-8") as log:
            log.write(traceback.format_exc())
        raise
    finally:
        write_json(directory / "run_summary.json", row)
        append_registry(row)
        if row["status"] != "completed":
            try:
                from src.results_registry import rebuild

                rebuild(ROOT)
            except Exception:
                with (directory / "logs/index_error.log").open("a", encoding="utf-8") as log:
                    log.write(traceback.format_exc())
    from src.results_registry import rebuild

    rebuild(ROOT)
    return row
