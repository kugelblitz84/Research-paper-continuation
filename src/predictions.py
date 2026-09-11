"""Split-neutral endpoint prediction serialization shared by validation and evaluation."""

import pandas as pd
import torch

from src.config import CLASSES, TASK_CLASSES
from src.metrics import classification_metrics
from src.routing import endpoint_output, hard_route, oracle_route


@torch.inference_mode()
def prediction_frame(model, loader, system_type, device):
    model.eval()
    rows = []
    for batch in loader:
        out = model(batch["image"].to(device))
        truth = batch["target"]
        if system_type == "flat":
            logits = out.float().cpu()
            probabilities = logits.softmax(1)
            pred = logits.argmax(1)
        else:
            s1, s2 = out["task1"].float().cpu(), out["task2"].float().cpu()
            p1, p2 = s1.softmax(1), s2.softmax(1)
            pred, probabilities = endpoint_output({"task1": s1, "task2": s2}, system_type)
            oracle = oracle_route(truth, s2)
        for i, image_id in enumerate(batch["image_id"]):
            y, pr = int(truth[i]), int(pred[i])
            row = dict(
                image_id=image_id,
                true_class=y,
                true_label=CLASSES[y],
                predicted_class=pr,
                predicted_label=CLASSES[pr],
            )
            if system_type == "flat":
                for c in range(4):
                    row.update(
                        {
                            f"logit_{c}": float(logits[i, c]),
                            f"probability_{c}": float(probabilities[i, c]),
                        }
                    )
            else:
                row.update(
                    stage1_truth=int(y > 0),
                    stage1_prediction=int(s1[i].argmax()),
                    stage2_truth=y - 1 if y > 0 else None,
                    stage2_prediction=int(s2[i].argmax()),
                    predicted_hard_gate=int(hard_route(s1[i : i + 1], s2[i : i + 1])[0]),
                    routing_mode="soft" if system_type.endswith("soft") else "hard",
                    oracle_gate=int(oracle[i]),
                )
                for stage, ls, ps in ((1, s1, p1), (2, s2, p2)):
                    for c in range(ls.shape[1]):
                        row.update(
                            {
                                f"stage{stage}_logit_{c}": float(ls[i, c]),
                                f"stage{stage}_probability_{c}": float(ps[i, c]),
                            }
                        )
                # All task heads are computed, including Stage2 on true NM; truth remains missing.
                row["stage2_used_by_hard_gate"] = bool(s1[i].argmax() == 1)
            for c in range(4):
                row[f"probability_{c}"] = float(probabilities[i, c])
            rows.append(row)
    frame = pd.DataFrame(rows)
    if frame.empty or frame.image_id.duplicated().any():
        raise ValueError("Empty or duplicate prediction IDs")
    return frame


def prediction_metrics(frame, system_type):
    metrics = {"endpoint": classification_metrics(frame.true_class, frame.predicted_class)}
    if system_type != "flat":
        metrics["oracle"] = classification_metrics(frame.true_class, frame.oracle_gate)
        malignant = frame.true_class > 0
        metrics["stage1"] = classification_metrics(
            frame.stage1_truth, frame.stage1_prediction, TASK_CLASSES[0]
        )
        if malignant.any():
            metrics["stage2_malignant_subset"] = classification_metrics(
                frame.loc[malignant, "stage2_truth"].astype(int),
                frame.loc[malignant, "stage2_prediction"],
                TASK_CLASSES[1],
            )
        metrics["routing"] = dict(
            malignant_false_negative=int(((frame.stage1_prediction == 0) & malignant).sum()),
            nonmalignant_false_positive=int(((frame.stage1_prediction == 1) & ~malignant).sum()),
            malignant_sensitivity=float((frame.loc[malignant, "stage1_prediction"] == 1).mean())
            if malignant.any()
            else None,
            oracle_gap=metrics["oracle"]["macro_f1"] - metrics["endpoint"]["macro_f1"],
        )
    return metrics
