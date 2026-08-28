import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


def classification_metrics(y_true, y_pred, y_prob=None):

    results = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0
        ),
        "recall": recall_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0
        ),
        "f1": f1_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0
        )
    }

    if y_prob is not None:
        try:
            if len(np.unique(y_true)) == 2:
                results["auc"] = roc_auc_score(
                    y_true,
                    y_prob
                )
        except Exception:
            results["auc"] = None

    return results


def binary_confidence(probabilities):

    probabilities = np.asarray(probabilities)

    if probabilities.ndim == 1:
        return np.maximum(probabilities, 1 - probabilities)

    return np.max(probabilities, axis=1) 
