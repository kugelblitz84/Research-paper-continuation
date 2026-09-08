"""Fixed-label confusion-derived statistics without rounding intermediate values."""

import numpy as np

from src.config import CLASSES


def classification_metrics(truth, prediction, classes=CLASSES):
    y, p = np.asarray(truth), np.asarray(prediction)
    k = len(classes)
    if (
        y.ndim != 1
        or p.shape != y.shape
        or not y.size
        or not np.isin(y, range(k)).all()
        or not np.isin(p, range(k)).all()
    ):
        raise ValueError("Invalid labels, shape, or empty predictions")
    cm = np.bincount(k * y.astype(int) + p.astype(int), minlength=k * k).reshape(k, k)
    support = cm.sum(1)
    precision = np.divide(cm.diagonal(), cm.sum(0), out=np.zeros(k), where=cm.sum(0) > 0)
    recall = np.divide(cm.diagonal(), support, out=np.zeros(k), where=support > 0)
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros(k),
        where=precision + recall > 0,
    )
    return dict(
        sample_count=len(y),
        accuracy=float(cm.trace() / len(y)),
        balanced_accuracy=float(recall[support > 0].mean()),
        macro_f1=float(f1.mean()),
        weighted_f1=float((f1 * support).sum() / len(y)),
        macro_precision=float(precision.mean()),
        macro_recall=float(recall.mean()),
        confusion_matrix=cm.tolist(),
        class_names=list(classes),
        per_class={
            n: dict(
                precision=float(precision[i]),
                recall=float(recall[i]),
                f1=float(f1[i]),
                support=int(support[i]),
            )
            for i, n in enumerate(classes)
        },
    )
