import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from src.metrics import classification_metrics
from src.statistics import paired_inputs, paired_statistics


def test_metrics_against_independent_sklearn():
    y = np.array([0, 0, 0, 1, 1, 2, 2, 3])
    p = np.array([0, 0, 1, 1, 2, 2, 3, 0])
    m = classification_metrics(y, p)
    assert m["macro_f1"] == pytest.approx(f1_score(y, p, labels=[0, 1, 2, 3], average="macro"))
    assert m["balanced_accuracy"] == pytest.approx(balanced_accuracy_score(y, p))
    assert m["accuracy"] == accuracy_score(y, p)
    assert m["per_class"]["scc"]["support"] == 1


def test_bootstrap_exact_paired_known_case():
    f = pd.DataFrame(
        dict(image_id=list("abcd"), true_class=[0, 1, 2, 3], predicted_class=[0, 1, 2, 3])
    )
    h = f.copy()
    h["predicted_class"] = [1, 2, 3, 0]
    result, boot = paired_statistics(f, h.iloc[::-1], replicates=20)
    assert result["differences"]["macro_f1"] == -1
    assert result["intervals"]["difference_macro_f1"] == [-1, -1]
    assert result["mcnemar"]["exact_two_sided_p"] == 0.125
    assert len(boot) == 20
    identical, _ = paired_statistics(f, f, replicates=2)
    assert identical["mcnemar"]["exact_two_sided_p"] == 1
    assert identical["intervals"]["difference_macro_f1"] == [0, 0]


def test_pairing_failures():
    f = pd.DataFrame(
        dict(image_id=list("abcd"), true_class=[0, 1, 2, 3], predicted_class=[0, 1, 2, 3])
    )
    h = f.copy()
    h.loc[0, "image_id"] = "b"
    with pytest.raises(ValueError):
        paired_inputs(f, h)
    h = f.copy()
    h.loc[0, "true_class"] = 1
    with pytest.raises(ValueError):
        paired_inputs(f, h)
    with pytest.raises(ValueError):
        paired_inputs(f, f.iloc[:-1])
    with pytest.raises(ValueError):
        classification_metrics([0.5], [0])
