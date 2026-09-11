"""Final saved-result reporting for all 35 cells; never selects models or runs inference."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import CLASSES, ROOT, digest
from src.experiment_catalog import BACKBONE_LABELS, SYSTEMS, expected_cells
from src.results_registry import RESULT_COLUMNS, _merge, atomic_csv, collect, read_csv


def final_tables(root=ROOT, *, split="test", seed=42, result_choices=None):
    if split not in ("validation", "test", "external"):
        raise ValueError("Choose one explicit split")
    root, choices = Path(root), result_choices or {}
    discovered, _, _ = collect(root)
    results = _merge(
        read_csv(root / "results/master_results.csv"), discovered, "result_id", scientific=True
    )
    selected, coverage = [], []
    for cell in expected_cells(seed):
        candidates = [
            r
            for r in results
            if r["run_id"] == cell["run_id"]
            and r["split"] == split
            and r["evaluation_mode"] == "endpoint"
            and r["status"] == "COMPLETED"
            and r["system_type"] == cell["system_type"]
            and r["backbone"] == cell["backbone"]
            and str(r["seed"]) == str(seed)
        ]
        if cell["run_id"] in choices:
            candidates = [r for r in candidates if r["result_id"] == choices[cell["run_id"]]]
            if len(candidates) != 1:
                raise ValueError(f"Unknown result choice for {cell['run_id']}")
        state = "MISSING" if not candidates else "AMBIGUOUS" if len(candidates) > 1 else "AVAILABLE"
        if state == "AVAILABLE":
            row = candidates[0]
            required = (
                "config_hash",
                "checkpoint_sha256",
                "prediction_sha256",
                "manifest_hash_isic",
                "sample_count",
            )
            if any(row.get(k) in ("", None) for k in required):
                state = "INCOMPLETE_PROVENANCE"
            else:
                selected.append(row)
        coverage.append(
            dict(
                cell,
                split=split,
                status=state,
                candidate_result_ids=";".join(r["result_id"] for r in candidates),
            )
        )
    frame = pd.DataFrame(selected, columns=RESULT_COLUMNS)
    if selected:
        if len({(r["manifest_hash_isic"], int(r["sample_count"])) for r in selected}) != 1:
            raise ValueError("Results do not share a matched manifest and sample count")
    columns = [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "weighted_f1",
        *[f"{c}_{m}" for c in CLASSES for m in ("precision", "recall", "f1", "support")],
    ]
    matrices = {}
    for metric in columns + ["oracle_gap", "malignant_sensitivity"]:
        matrix = pd.DataFrame(
            np.nan, index=list(BACKBONE_LABELS.values()), columns=[s[0] for s in SYSTEMS.values()]
        )
        matrix.index.name = "Backbone"
        for row in selected:
            value = row.get(metric)
            if value not in ("", None):
                value = float(value)
                if not np.isfinite(value) or (
                    not metric.endswith("support")
                    and not (-1 <= value <= 1 if metric == "oracle_gap" else 0 <= value <= 1)
                ):
                    raise ValueError(f"Invalid {metric}")
                matrix.loc[BACKBONE_LABELS[row["backbone"]], SYSTEMS[row["system_type"]][0]] = value
        matrices[metric] = matrix
    summary, deltas = [], []
    primary = matrices["macro_f1"]
    for system, (code, _) in SYSTEMS.items():
        values = primary[code]
        summary.append(
            dict(
                system_type=system,
                system_code=code,
                n=int(values.count()),
                mean=values.mean(),
                median=values.median(),
                status="COMPLETE" if values.count() == 7 else "INCOMPLETE",
            )
        )
        for backbone in primary.index:
            value = primary.loc[backbone, code]
            deltas.append(
                dict(
                    backbone=backbone,
                    system_code=code,
                    macro_f1=value,
                    delta_vs_flat=value - primary.loc[backbone, "F"],
                    delta_vs_h1=value - primary.loc[backbone, "H1"],
                )
            )
    return dict(
        split=split,
        coverage=pd.DataFrame(coverage),
        results=frame,
        matrices=matrices,
        summary=pd.DataFrame(summary),
        deltas=pd.DataFrame(deltas),
        status="COMPLETE" if len(selected) == 35 else "INCOMPLETE",
        selected_count=len(selected),
    )


def _heatmap(ax, matrix, title, *, delta=False):
    values = matrix.to_numpy(dtype=float)
    cmap = plt.get_cmap("RdBu_r" if delta else "viridis").with_extremes(bad="#eeeeee")
    limit = (
        max(0.05, float(np.nanmax(np.abs(values)))) if delta and np.isfinite(values).any() else 1
    )
    image = ax.imshow(
        np.ma.masked_invalid(values),
        cmap=cmap,
        vmin=-limit if delta else 0,
        vmax=limit if delta else 1,
        aspect="auto",
    )
    ax.set(
        xticks=range(len(matrix.columns)),
        xticklabels=matrix.columns,
        yticks=range(len(matrix.index)),
        yticklabels=matrix.index,
        title=title,
    )
    for i in range(len(matrix.index)):
        for j in range(len(matrix.columns)):
            value = values[i, j]
            label = "missing" if np.isnan(value) else f"{value:+.3f}" if delta else f"{value:.3f}"
            color = "black" if np.isnan(value) or (not delta and value > 0.55) else "white"
            if delta:
                color = "black" if np.isnan(value) or abs(value) < limit * 0.5 else "white"
            ax.text(j, i, label, ha="center", va="center", fontsize=8, color=color)
    ax.figure.colorbar(image, ax=ax, fraction=0.045, pad=0.03)


def overview_figure(report):
    """One composite publication figure containing the full 7x5 endpoint grid."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), layout="constrained")
    primary = report["matrices"]["macro_f1"]
    _heatmap(axes[0, 0], primary, "A  Endpoint macro-F1")
    _heatmap(axes[0, 1], report["matrices"]["balanced_accuracy"], "B  Balanced accuracy")
    _heatmap(axes[1, 0], report["matrices"]["scc_f1"], "C  SCC class F1")
    delta = primary.drop(columns="F").subtract(primary.F, axis=0)
    _heatmap(axes[1, 1], delta, "D  Matched macro-F1 difference versus Flat", delta=True)
    fig.suptitle(
        f"{report['split'].title()} endpoint evidence: {report['selected_count']}/35 cells "
        f"({report['status']})\nF Flat | H1 Shared-Hard | H2 Shared-Soft | "
        "H3 Dedicated-Hard | H4 Dedicated-Soft",
        fontsize=12,
    )
    return fig


def classwise_figure(report):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), layout="constrained")
    for ax, cls in zip(axes.flat, CLASSES):
        _heatmap(ax, report["matrices"][f"{cls}_f1"], cls.replace("_", " ").title() + " F1")
    fig.suptitle(f"{report['split'].title()} per-class F1 â€” {report['status']}")
    return fig


def confusion_figure(report):
    fig, axes = plt.subplots(7, 5, figsize=(15, 19), layout="constrained")
    lookup = {(r["backbone"], r["system_type"]): r for r in report["results"].to_dict("records")}
    for i, (backbone, label) in enumerate(BACKBONE_LABELS.items()):
        for j, (system, (code, _)) in enumerate(SYSTEMS.items()):
            ax = axes[i, j]
            row = lookup.get((backbone, system), {})
            raw = row.get("confusion_matrix")
            cm = np.asarray(json.loads(raw), dtype=float) if raw else np.empty(0)
            ax.set_title(f"{label}\n{code}", fontsize=9)
            if cm.shape != (4, 4):
                ax.text(0.5, 0.5, "missing", ha="center", va="center")
                ax.set_axis_off()
                continue
            if not np.isfinite(cm).all() or (cm < 0).any():
                raise ValueError("Invalid confusion matrix")
            normalized = np.divide(
                cm, cm.sum(1)[:, None], out=np.zeros_like(cm), where=cm.sum(1)[:, None] > 0
            )
            ax.imshow(normalized, vmin=0, vmax=1, cmap="Blues")
            for y in range(4):
                for x in range(4):
                    ax.text(
                        x,
                        y,
                        f"{int(cm[y, x])}",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="white" if normalized[y, x] > 0.5 else "black",
                    )
            ax.set(
                xticks=range(4),
                yticks=range(4),
                xticklabels=["NM", "MEL", "BCC", "SCC"],
                yticklabels=["NM", "MEL", "BCC", "SCC"],
            )
            ax.tick_params(labelsize=6)
    fig.suptitle(
        f"{report['split'].title()} confusion matrices: rows truth, columns prediction\n"
        f"Row-normalized color; annotations are counts â€” {report['status']}"
    )
    return fig


def export_report(report, root=ROOT):
    """Create a content-addressed evidence bundle; never overwrite a previous report."""
    records = report["results"].fillna("").to_dict("records")
    key = digest(
        dict(
            version=1,
            split=report["split"],
            results=records,
            coverage=report["coverage"].to_dict("records"),
        )
    )
    directory = Path(root) / "results/final" / f"{report['split']}_{key[:16]}"
    directory.mkdir(parents=True, exist_ok=True)
    manifest = directory / "report_manifest.json"
    if manifest.exists():
        from src.reproducibility import sha256

        saved = json.loads(manifest.read_text())
        for name, expected in saved["files"].items():
            if sha256(directory / name) != expected:
                raise ValueError(f"Exported artifact changed: {name}")
        return directory
    tables = {k: report[k] for k in ("coverage", "results", "summary", "deltas")}
    tables.update({k + "_matrix": v.reset_index() for k, v in report["matrices"].items()})
    for name, frame in tables.items():
        atomic_csv(
            directory / f"{name}.csv",
            frame.astype(object).where(frame.notna(), "").to_dict("records"),
            list(frame),
        )
    for name, draw in [
        ("overview_35_cells", overview_figure),
        ("per_class_f1", classwise_figure),
        ("confusion_matrices", confusion_figure),
    ]:
        figure = draw(report)
        for extension in ("png", "svg", "pdf"):
            figure.savefig(directory / f"{name}.{extension}", dpi=300, bbox_inches="tight")
        plt.close(figure)
    caption = (
        f"{report['split'].title()} endpoint results for five systems and seven backbones. "
        f"{report['selected_count']}/35 cells available; report status {report['status']}. "
        "Missing cells are grey and excluded from averages. Means and medians are descriptive "
        "across available backbones, not independent-replicate uncertainty estimates. "
        "Differences are matched by backbone. Test results do not determine Block-A selection. "
        "H1 retains its historical auxiliary third task and selection rule. "
        "No confidence intervals or significance claims are inferred from this illustration.\n"
    )
    (directory / "figure_caption.txt").write_text(caption, encoding="utf-8")
    from src.artifacts import write_json
    from src.reproducibility import sha256

    write_json(
        manifest,
        dict(
            report_digest=key,
            split=report["split"],
            status=report["status"],
            result_ids=[r["result_id"] for r in records],
            files={p.name: sha256(p) for p in directory.iterdir() if p.is_file()},
        ),
    )
    return directory
