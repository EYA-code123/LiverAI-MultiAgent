import numpy as np

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


def classification_metrics(
    y_true,
    y_pred,
    y_prob=None
):

    results = {

        "accuracy":
            float(
                accuracy_score(
                    y_true,
                    y_pred
                )
            ),

        "balanced_accuracy":
            float(
                balanced_accuracy_score(
                    y_true,
                    y_pred
                )
            ),

        "precision":
            float(
                precision_score(
                    y_true,
                    y_pred,
                    average="weighted",
                    zero_division=0
                )
            ),

        "recall":
            float(
                recall_score(
                    y_true,
                    y_pred,
                    average="weighted",
                    zero_division=0
                )
            ),

        "f1":
            float(
                f1_score(
                    y_true,
                    y_pred,
                    average="weighted",
                    zero_division=0
                )
            ),

        "confusion_matrix":
            confusion_matrix(
                y_true,
                y_pred
            ).tolist()
    }

    # ---------------------------------------------------------
    # AUC
    # ---------------------------------------------------------

    if y_prob is not None:

        try:

            y_prob = np.asarray(
                y_prob
            )

            classes = np.unique(
                y_true
            )

            if len(classes) == 2:

                if y_prob.ndim == 1:

                    results["auc"] = float(

                        roc_auc_score(
                            y_true,
                            y_prob
                        )
                    )

                else:

                    results["auc"] = float(

                        roc_auc_score(
                            y_true,
                            y_prob[:, 1]
                        )
                    )

            elif y_prob.ndim == 2:

                results["auc"] = float(

                    roc_auc_score(

                        y_true,

                        y_prob,

                        multi_class="ovr",

                        average="weighted"
                    )
                )

        except Exception:

            results["auc"] = None

    else:

        results["auc"] = None

    return results


def coordination_metrics(
    agent_results,
    conflicts,
    decision
):

    total = len(
        agent_results
    )

    valid = [

        result

        for result in agent_results

        if result.get(
            "prediction"
        ) is not None
    ]

    coverage = (

        len(valid)
        /
        total

        if total > 0

        else 0.0
    )

    conflict_rate = (

        len(conflicts)
        /
        max(
            len(valid),
            1
        )
    )

    if valid:

        mean_trust = np.mean([

            float(
                r.get(
                    "trust",
                    0.0
                )
            )

            for r in valid
        ])

        mean_confidence = np.mean([

            float(
                r.get(
                    "confidence",
                    0.0
                )
            )

            for r in valid
        ])

        mean_quality = np.mean([

            float(
                r.get(
                    "quality",
                    0.0
                )
            )

            for r in valid
        ])

    else:

        mean_trust = 0.0

        mean_confidence = 0.0

        mean_quality = 0.0

    return {

        "agent_coverage":
            float(
                coverage
            ),

        "conflict_rate":
            float(
                conflict_rate
            ),

        "mean_trust":
            float(
                mean_trust
            ),

        "mean_confidence":
            float(
                mean_confidence
            ),

        "mean_quality":
            float(
                mean_quality
            ),

        "decision_confidence":
            float(
                decision.get(
                    "confidence",
                    0.0
                )
            ),

        "decision_risk":
            float(
                decision.get(
                    "risk_score",
                    1.0
                )
            )
    }
