"""Block-A endpoint validation aggregation. Test/oracle evidence never enters ranking."""

import math

import pandas as pd

from src.experiment_catalog import BACKBONE_LABELS, SYSTEMS


def aggregate(status):
    expected = {(b, s) for b in BACKBONE_LABELS for s in SYSTEMS}
    if set(zip(status.backbone, status.system_type)) != expected or len(status) != 35:
        raise ValueError("Exactly 35 unique matched cells required")
    table = pd.DataFrame(
        index=list(BACKBONE_LABELS.values()), columns=[v[1] for v in SYSTEMS.values()], dtype=float
    )
    for row in status.to_dict("records"):
        value = row.get("validation_macro_f1")
        if row["status"] == "COMPLETED" and value not in ("", None) and not pd.isna(value):
            value = float(value)
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("Invalid endpoint validation macro-F1")
            table.loc[BACKBONE_LABELS[row["backbone"]], SYSTEMS[row["system_type"]][1]] = value
    summary = []
    complete = bool(table.notna().all().all())
    for system, (code, column) in SYSTEMS.items():
        values = table[column]
        flat_delta, h1_delta = values - table.Flat, values - table.H1_Shared_Hard
        summary.append(
            dict(
                system_type=system,
                system_code=code,
                count=int(values.count()),
                status="COMPLETED" if values.count() == 7 else "INCOMPLETE",
                mean=values.mean(),
                median=values.median(),
                delta_vs_flat=flat_delta.mean(),
                flat_matched_count=int(flat_delta.count()),
                delta_vs_h1=h1_delta.mean(),
                h1_matched_count=int(h1_delta.count()),
            )
        )
    summary = pd.DataFrame(summary)
    selection = {
        "status": "INCOMPLETE",
        "advancing": [],
        "selected_system": None,
        "metric": "mean validation endpoint macro_f1",
        "near_tie_tolerance": 0.005,
        "required_cells": 35,
        "available_cells": int(table.notna().sum().sum()),
    }
    if complete:
        ranked = summary[summary.system_type != "flat"].sort_values(
            ["mean", "system_code"], ascending=[False, True]
        )
        first, second = ranked.iloc[0], ranked.iloc[1]
        gap = float(first["mean"] - second["mean"])
        tie = gap <= 0.005 or math.isclose(gap, 0.005, rel_tol=0, abs_tol=1e-12)
        selection.update(
            status="NEAR_TIE" if tie else "SELECTED",
            advancing=[first.system_type, second.system_type] if tie else [first.system_type],
            selected_system=None if tie else first.system_type,
            gap=gap,
        )
    table.loc["Mean"] = table.mean()
    # Compute median over backbone rows only (exclude the newly added Mean).
    table.loc["Median"] = table.iloc[:7].median()
    table.index.name = "Backbone"
    return table.reset_index(), summary, selection
