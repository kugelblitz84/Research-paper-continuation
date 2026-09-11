"""Explicit final test execution, separate from development and saved-result reporting."""

from pathlib import Path

import pandas as pd

from src.config import ROOT
from src.experiment_catalog import expected_cells
from src.results_registry import _merge, collect, matrix_status, read_csv


def evaluation_plan(root=ROOT, seed=42):
    root = Path(root)
    new, _, evidence = collect(root)
    rows = _merge(read_csv(root / "results/master_results.csv"), new, "result_id", scientific=True)
    plan = []
    for cell in expected_cells(seed):
        item = evidence.get(cell["run_id"], {})
        frozen = item.get("frozen", {})
        best = frozen.get("best", {})
        saved = [
            r
            for r in rows
            if r["run_id"] == cell["run_id"]
            and r["split"] == "test"
            and r["evaluation_mode"] == "endpoint"
            and r["status"] == "COMPLETED"
        ]
        matched = [
            r
            for r in saved
            if frozen
            and r.get("config_hash") == frozen.get("config_hash")
            and r.get("checkpoint_sha256") == best.get("sha256")
        ]
        if matched:
            state = "SAVED"
        elif saved:
            state = "REVIEW_EXISTING_EVIDENCE"
        elif item.get("summary", {}).get("status") != "completed":
            state = "TRAINING_INCOMPLETE"
        elif not frozen or not best.get("path") or not (root / best["path"]).is_file():
            state = "VM_CHECKPOINT_REQUIRED"
        else:
            state = "READY_AFTER_DEVELOPMENT_FREEZE"
        plan.append(
            dict(
                cell,
                evaluation_status=state,
                saved_result_ids=";".join(r["result_id"] for r in saved),
            )
        )
    return pd.DataFrame(plan)


def evaluate_final(
    run_ids,
    *,
    run_evaluation=False,
    development_frozen=False,
    root=ROOT,
    seed=42,
    data_root=None,
    device="cuda",
):
    """No default inference; existing evidence is never silently regenerated."""
    plan = evaluation_plan(root, seed)
    if not run_evaluation:
        return plan
    if not development_frozen:
        raise ValueError("Explicit development freeze is required before final test execution")
    status = matrix_status(root, seed)
    if not status.status.eq("COMPLETED").all() or status.validation_result_id.eq("").any():
        raise ValueError("Complete all 35 training/validation cells before final test execution")
    allowed = set(plan.run_id)
    if not run_ids or len(run_ids) != len(set(run_ids)) or not set(run_ids) <= allowed:
        raise ValueError("Choose unique, explicit run IDs from the 35-cell matrix")
    chosen = plan.set_index("run_id").loc[run_ids]
    if not chosen.evaluation_status.isin(["SAVED", "READY_AFTER_DEVELOPMENT_FREEZE"]).all():
        raise ValueError("Restore/review existing evidence and checkpoints before evaluating")
    if Path(root).resolve() != ROOT.resolve():
        raise ValueError("Real evaluation must use the configured repository root")
    from src.evaluation import evaluate

    outcomes = []
    for run_id, row in chosen.iterrows():
        if row.evaluation_status == "SAVED":
            outcomes.append(dict(run_id=run_id, status="SKIPPED_SAVED"))
        else:
            outcomes.append(evaluate(run_id, split="test", data_root=data_root, device=device))
    return outcomes
