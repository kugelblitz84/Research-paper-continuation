"""Replay stored legacy evidence; these are NOT newly trained model results."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.metrics import classification_metrics
from src.reproducibility import sha256
from src.statistics import paired_statistics


def main():
    csv = ROOT / "results/historical/b0_paired_internal_test_predictions.csv"
    old = ROOT / "results/historical/b0_original_statistics.json"
    f = pd.read_csv(csv)
    reference = json.loads(old.read_text())
    flat = pd.DataFrame(
        dict(
            image_id=f.sample_id,
            true_class=f.true_label,
            predicted_class=f.flat_prediction,
        )
    )
    shared = pd.DataFrame(
        dict(
            image_id=f.sample_id,
            true_class=f.true_label,
            predicted_class=f.shared_predicted_gate,
        )
    )
    result, boot = paired_statistics(flat, shared)
    for metric in ("accuracy", "macro_f1"):
        expected = reference["paired_differences"][metric]
        assert abs(result["differences"][metric] - expected["delta_hierarchy_minus_flat"]) < 1e-12
        assert np.allclose(
            result["intervals"][f"difference_{metric}"],
            expected["paired_bootstrap_95ci"],
            atol=1e-12,
            rtol=0,
        )
    assert (
        result["mcnemar"]["exact_two_sided_p"]
        == reference["mcnemar_exact"]["exact_two_sided_p_value"]
    )
    result["oracle"] = classification_metrics(f.true_label, f.shared_oracle_gate)
    result["evidence_kind"] = "historical_prediction_replay_not_retraining"
    result["status"] = "PASS"
    result["inputs"] = {
        "predictions_sha256": sha256(csv),
        "historical_statistics_sha256": sha256(old),
    }
    (ROOT / "validation/historical_statistics_replay.json").write_text(json.dumps(result, indent=2))
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "status",
                    "evidence_kind",
                    "differences",
                    "intervals",
                    "mcnemar",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
