# =============================================================================
# LiverAI-MultiAgent
# CONFLICT DETECTOR
# =============================================================================

import numpy as np


class ConflictDetector:

    def __init__(
        self,
        confidence_threshold=0.20
    ):

        self.confidence_threshold = (
            confidence_threshold
        )

    # -------------------------------------------------------------------------
    # DETECT CONFLICTS
    # -------------------------------------------------------------------------

    def detect(
        self,
        results
    ):

        conflicts = []

        valid_results = [
            r for r in results
            if getattr(r, "success", False)
        ]

        groups = {}

        for result in valid_results:

            task_type = getattr(
                result,
                "task_type",
                "unknown"
            )

            groups.setdefault(
                task_type,
                []
            )

            groups[
                task_type
            ].append(result)

        # ---------------------------------------------------------------------
        # COMPARE ONLY SAME TASK
        # ---------------------------------------------------------------------

        for task_type, task_results in groups.items():

            if len(task_results) < 2:
                continue

            for i in range(
                len(task_results)
            ):

                for j in range(
                    i + 1,
                    len(task_results)
                ):

                    a = task_results[i]
                    b = task_results[j]

                    if (
                        a.prediction
                        ==
                        b.prediction
                    ):
                        continue

                    confidence_gap = abs(
                        float(a.confidence)
                        -
                        float(b.confidence)
                    )

                    conflicts.append({

                        "task_type":
                            task_type,

                        "agent_a":
                            a.agent_id,

                        "agent_b":
                            b.agent_id,

                        "prediction_a":
                            a.prediction,

                        "prediction_b":
                            b.prediction,

                        "confidence_a":
                            float(
                                a.confidence
                            ),

                        "confidence_b":
                            float(
                                b.confidence
                            ),

                        "trust_a":
                            float(
                                a.trust
                            ),

                        "trust_b":
                            float(
                                b.trust
                            ),

                        "confidence_gap":
                            float(
                                confidence_gap
                            ),

                        "severity":
                            self._severity(
                                confidence_gap
                            )
                    })

        return conflicts

    # -------------------------------------------------------------------------
    # SEVERITY
    # -------------------------------------------------------------------------

    def _severity(
        self,
        confidence_gap
    ):

        if confidence_gap < 0.20:
            return "low"

        if confidence_gap < 0.40:
            return "medium"

        return "high"

    # -------------------------------------------------------------------------
    # AGREEMENT
    # -------------------------------------------------------------------------

    def agreement_score(
        self,
        results
    ):

        valid_results = [
            r for r in results
            if getattr(r, "success", False)
        ]

        if len(valid_results) <= 1:

            return 1.0

        predictions = [
            str(r.prediction)
            for r in valid_results
        ]

        counts = {}

        for prediction in predictions:

            counts[prediction] = (
                counts.get(
                    prediction,
                    0
                ) + 1
            )

        maximum = max(
            counts.values()
        )

        return float(
            np.clip(
                maximum
                /
                len(predictions),
                0.0,
                1.0
            )
        )
