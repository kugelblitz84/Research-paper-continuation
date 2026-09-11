"""One explicit training entry point with registry-aware overwrite protection."""

import argparse

from src.config import ROOT, digest
from src.experiment_catalog import BACKBONE_LABELS, SYSTEMS, experiment_config
from src.results_registry import run_evidence


def disposition(config, *, root=ROOT, resume=False):
    """Fail closed on incomplete provenance; never recreate registry-only completed runs."""
    run_id = config["experiment_id"]
    item = run_evidence(root).get(run_id)
    if not item:
        if resume:
            raise FileNotFoundError("Resume run not found")
        return "RUN"
    saved = item.get("config", {})
    frozen = item.get("frozen", {})
    actual = frozen.get("config_hash") or (digest(saved) if saved else None)
    if actual and actual != digest(config):
        raise ValueError(f"Incompatible config for {run_id}")
    for event in item["events"] + [item.get("summary", {})]:
        for key, expected in [
            ("system_type", config["system_type"]),
            ("architecture", config["architecture"]),
            ("seed", config["seed"]),
        ]:
            if event.get(key) not in (None, "") and str(event[key]) != str(expected):
                raise ValueError(f"Incompatible identity for {run_id}: {key}")
    completed = any(
        e.get("status") == "completed" for e in item["events"] + [item.get("summary", {})]
    )
    if completed:
        if resume:
            raise ValueError("Completed experiment cannot be resumed or overwritten")
        return "SKIP" if actual else "SKIP_MISSING_PROVENANCE"
    if item.get("active"):
        return "RUNNING"
    if resume:
        if not saved or not item["directory_present"]:
            raise ValueError("Resume requires canonical config and checkpoint artifacts")
        return "RESUME"
    return (
        "FAILED"
        if item.get("summary", {}).get("status") in ("failed", "interrupted")
        else "INCOMPLETE"
    )


def run_experiment(
    system, backbone, seed=42, *, resume=False, data_root=None, device="cuda", allow_training=False
):
    config = experiment_config(system, backbone, seed)
    action = disposition(config, resume=resume)
    if action not in ("RUN", "RESUME"):
        return {"run_id": config["experiment_id"], "status": action}
    from src.training import train

    result = train(
        config, allow_training=allow_training, resume=resume, data_root=data_root, device=device
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", choices=SYSTEMS, required=True)
    parser.add_argument("--backbone", choices=BACKBONE_LABELS, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data-root")
    args = parser.parse_args()
    print(
        run_experiment(
            args.system,
            args.backbone,
            args.seed,
            resume=args.resume,
            device=args.device,
            data_root=args.data_root,
            allow_training=True,
        )
    )


if __name__ == "__main__":
    main()
