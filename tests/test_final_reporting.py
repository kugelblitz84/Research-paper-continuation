"""Synthetic final-report tests: no actual model, image or test inference."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import pytest

from src import final_evaluation, final_reporting
from src.config import CLASSES
from src.experiment_catalog import expected_cells
from src.results_registry import RESULT_COLUMNS


def synthetic_rows():
    rows = []
    for i, cell in enumerate(expected_cells()):
        row = {k: "" for k in RESULT_COLUMNS}
        row.update(
            cell,
            split="test",
            evaluation_mode="endpoint",
            status="COMPLETED",
            result_id=f"synthetic-{i}",
            config_hash="synthetic-config",
            checkpoint_sha256=f"synthetic-checkpoint-{i}",
            prediction_sha256=f"synthetic-prediction-{i}",
            manifest_hash_isic="synthetic-manifest",
            sample_count=8,
            macro_f1=0.6 + (i % 5) * 0.01,
            balanced_accuracy=0.7,
            confusion_matrix="[[2,0,0,0],[0,2,0,0],[0,0,2,0],[0,0,0,2]]",
        )
        for cls in CLASSES:
            row[f"{cls}_f1"] = 0.7
        rows.append(row)
    return rows


def use_rows(monkeypatch, rows):
    monkeypatch.setattr(final_reporting, "collect", lambda root: (rows, [], {}))


def test_complete_35_report_and_export(tmp_path, monkeypatch):
    use_rows(monkeypatch, synthetic_rows())
    report = final_reporting.final_tables(tmp_path)
    assert report["status"] == "COMPLETE" and report["selected_count"] == 35
    assert report["matrices"]["macro_f1"].shape == (7, 5)
    assert report["summary"]["n"].eq(7).all()
    assert report["summary"].iloc[1]["mean"] == pytest.approx(0.61)
    assert report["deltas"].query('system_code == "H2"').delta_vs_flat.tolist() == pytest.approx(
        [0.02] * 7
    )
    output = final_reporting.export_report(report, tmp_path)
    assert (output / "overview_35_cells.svg").is_file()
    assert (output / "overview_35_cells.pdf").is_file()
    assert (output / "confusion_matrices.png").is_file()
    assert len(pd.read_csv(output / "results.csv")) == 35
    before = (output / "report_manifest.json").read_bytes()
    assert final_reporting.export_report(report, tmp_path) == output
    assert (output / "report_manifest.json").read_bytes() == before
    (output / "figure_caption.txt").write_text("tampered")
    with pytest.raises(ValueError, match="changed"):
        final_reporting.export_report(report, tmp_path)


def test_missing_test_does_not_use_validation_or_oracle(tmp_path, monkeypatch):
    rows = synthetic_rows()
    rows[0]["split"] = "validation"
    rows[1]["evaluation_mode"] = "oracle"
    use_rows(monkeypatch, rows)
    report = final_reporting.final_tables(tmp_path)
    assert report["selected_count"] == 33
    assert report["status"] == "INCOMPLETE"
    assert pd.isna(report["matrices"]["macro_f1"].iloc[0, 0])
    assert report["summary"].iloc[0]["n"] == 6


def test_ambiguous_needs_explicit_identity(tmp_path, monkeypatch):
    rows = synthetic_rows()
    duplicate = dict(
        rows[0], result_id="another", prediction_sha256="another-prediction", macro_f1=0.99
    )
    use_rows(monkeypatch, rows + [duplicate])
    report = final_reporting.final_tables(tmp_path)
    assert report["coverage"].iloc[0]["status"] == "AMBIGUOUS"
    assert pd.isna(report["matrices"]["macro_f1"].iloc[0, 0])
    chosen = final_reporting.final_tables(
        tmp_path, result_choices={rows[0]["run_id"]: rows[0]["result_id"]}
    )
    assert chosen["matrices"]["macro_f1"].iloc[0, 0] == 0.6
    with pytest.raises(ValueError, match="Unknown result"):
        final_reporting.final_tables(tmp_path, result_choices={rows[0]["run_id"]: "invalid"})


def test_missing_provenance_and_cohort_mismatch(tmp_path, monkeypatch):
    rows = synthetic_rows()
    rows[0]["checkpoint_sha256"] = ""
    use_rows(monkeypatch, rows)
    assert (
        final_reporting.final_tables(tmp_path)["coverage"].iloc[0]["status"]
        == "INCOMPLETE_PROVENANCE"
    )
    rows[1]["manifest_hash_isic"] = "other-cohort"
    with pytest.raises(ValueError, match="matched manifest"):
        final_reporting.final_tables(tmp_path)


def test_empty_report_draws_without_data(tmp_path, monkeypatch):
    use_rows(monkeypatch, [])
    report = final_reporting.final_tables(tmp_path)
    assert report["selected_count"] == 0
    assert report["matrices"]["macro_f1"].isna().all().all()
    assert set(report["coverage"].status) == {"MISSING"}
    for draw in (
        final_reporting.overview_figure,
        final_reporting.classwise_figure,
        final_reporting.confusion_figure,
    ):
        fig = draw(report)
        fig.canvas.draw()
        plt.close(fig)


def test_final_evaluation_default_and_completion_gate(tmp_path, monkeypatch):
    plan = final_evaluation.evaluate_final([], root=tmp_path)
    assert len(plan) == 35
    with pytest.raises(ValueError, match="development freeze"):
        final_evaluation.evaluate_final([], root=tmp_path, run_evaluation=True)
    with pytest.raises(ValueError, match="35 training/validation"):
        final_evaluation.evaluate_final(
            [], root=tmp_path, run_evaluation=True, development_frozen=True
        )


def test_final_evaluation_skips_saved(monkeypatch):
    from src import evaluation

    cells = expected_cells()
    run_id = cells[0]["run_id"]
    plan = pd.DataFrame([dict(c, evaluation_status="SAVED") for c in cells])
    status = pd.DataFrame(
        [dict(c, status="COMPLETED", validation_result_id="saved") for c in cells]
    )
    monkeypatch.setattr(final_evaluation, "evaluation_plan", lambda *a: plan)
    monkeypatch.setattr(final_evaluation, "matrix_status", lambda *a: status)
    monkeypatch.setattr(evaluation, "evaluate", lambda *a, **k: pytest.fail("inference called"))
    result = final_evaluation.evaluate_final([run_id], run_evaluation=True, development_frozen=True)
    assert result == [dict(run_id=run_id, status="SKIPPED_SAVED")]
