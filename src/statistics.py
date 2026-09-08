"""Paired image-level stratified bootstrap and exact secondary McNemar test."""

import json

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from src.artifacts import write_json
from src.config import CLASSES, ROOT
from src.metrics import classification_metrics
from src.reproducibility import sha256


def paired_inputs(flat, shared):
    required = {"image_id", "true_class", "predicted_class"}
    for frame in (flat, shared):
        if (
            not required <= set(frame)
            or frame.empty
            or frame.image_id.isna().any()
            or frame.image_id.duplicated().any()
        ):
            raise ValueError("Missing fields or duplicate/empty prediction IDs")
        classification_metrics(frame.true_class, frame.predicted_class)
    if set(flat.image_id) != set(shared.image_id):
        raise ValueError("Paired prediction ID sets differ")
    # Preserve frozen manifest ordering (Flat file) for historical seeded resampling.
    aligned = shared.set_index("image_id").loc[flat.image_id].reset_index()
    if not np.array_equal(flat.true_class.to_numpy(), aligned.true_class.to_numpy()):
        raise ValueError("Paired ground truth differs")
    return (
        flat.true_class.to_numpy(int),
        flat.predicted_class.to_numpy(int),
        aligned.predicted_class.to_numpy(int),
    )


def paired_statistics(flat, shared, *, replicates=10000, seed=42):
    y, f, h = paired_inputs(flat, shared)
    if replicates < 1:
        raise ValueError("Positive replicate count required")
    strata = [np.flatnonzero(y == i) for i in range(4)]
    if any(not len(s) for s in strata):
        raise ValueError("Every endpoint class must be present")
    rng = np.random.default_rng(seed)
    rows = []
    keys = ("accuracy", "balanced_accuracy", "macro_f1")
    for r in range(replicates):
        ix = np.concatenate([rng.choice(s, size=len(s), replace=True) for s in strata])
        fm, hm = (
            classification_metrics(y[ix], f[ix]),
            classification_metrics(y[ix], h[ix]),
        )
        row = {"replicate_id": r}
        for key in keys:
            row.update(
                {
                    f"flat_{key}": fm[key],
                    f"shared_hard_{key}": hm[key],
                    f"difference_{key}": hm[key] - fm[key],
                }
            )
        for c in CLASSES:
            row[f"difference_f1_{c}"] = hm["per_class"][c]["f1"] - fm["per_class"][c]["f1"]
        rows.append(row)
    boot = pd.DataFrame(rows)
    if not np.isfinite(boot.to_numpy()).all():
        raise ValueError("Nonfinite bootstrap replicate")
    fm, hm = classification_metrics(y, f), classification_metrics(y, h)
    intervals = {
        key: np.quantile(boot[key].to_numpy(np.float64), [0.025, 0.975], method="linear").tolist()
        for key in boot
        if key.startswith("difference")
    }
    fc, hc = f == y, h == y
    flat_only, shared_only = int((fc & ~hc).sum()), int((~fc & hc).sum())
    n = flat_only + shared_only
    result = dict(
        direction="shared_hard_minus_flat",
        bootstrap=dict(
            method="paired_truth_stratified",
            replicates=replicates,
            seed=seed,
            confidence=0.95,
            quantile_method="linear",
            unit="image",
        ),
        flat=fm,
        shared_hard=hm,
        differences={k: hm[k] - fm[k] for k in keys},
        intervals=intervals,
        mcnemar=dict(
            both_correct=int((fc & hc).sum()),
            both_wrong=int((~fc & ~hc).sum()),
            flat_only=flat_only,
            shared_only=shared_only,
            exact_two_sided_p=1.0 if not n else float(binomtest(flat_only, n, 0.5).pvalue),
        ),
    )
    return result, boot


def load_evaluation(evaluation_id):
    path = ROOT / "experiments/evaluations" / evaluation_id / "evaluation.json"
    meta = json.loads(path.read_text())
    csv = ROOT / meta["prediction_path"]
    if sha256(csv) != meta["prediction_sha256"]:
        raise ValueError("Predictions changed after evaluation")
    return meta, pd.read_csv(csv, float_precision="round_trip")


def compare_evaluations(flat_id, shared_id, config):
    fm, f = load_evaluation(flat_id)
    hm, h = load_evaluation(shared_id)
    if (
        fm["system_type"] != "flat"
        or hm["system_type"] != "shared_hard"
        or fm["architecture"] != hm["architecture"]
        or fm["split"] != hm["split"]
        or fm["manifest_hashes"] != hm["manifest_hashes"]
    ):
        raise ValueError("Comparison must use matched systems, backbone, split and manifests")
    p = config["statistics"]
    result, boot = paired_statistics(f, h, replicates=p["replicates"], seed=p["seed"])
    destination = ROOT / "results/tables" / f"{fm['architecture']}_{fm['split']}_comparison"
    destination.mkdir(parents=True, exist_ok=False)
    result["inputs"] = [fm, hm]
    write_json(destination / "statistics.json", result)
    boot.to_csv(destination / "bootstrap_replicates.csv", index=False, float_format="%.17g")
    rows = []
    history = (
        pd.read_csv(ROOT / "results/historical/document_reported_internal_test.csv")
        .set_index("architecture")
        .loc[fm["architecture"]]
    )
    for system, key in (("flat", "flat"), ("shared_hard", "shared_hard")):
        old = float(history[f"{system}_macro_f1"]) if fm["split"] == "test" else None
        score = result[key]["macro_f1"]
        rows.append(
            dict(
                system=system,
                historical_macro_f1=old,
                reconstructed_macro_f1=score,
                difference=None if old is None else score - old,
                provenance=history.provenance
                if old is not None
                else "No test reference used for validation",
            )
        )
    pd.DataFrame(rows).to_csv(destination / "historical_vs_reconstructed.csv", index=False)
    return result, destination
