# =============================================================================
# LiverAI-MultiAgent
# CONFLICT DETECTOR
# =============================================================================


class ConflictDetector:
    """
    Detects prediction conflicts between agents.

    Conflicts are ONLY evaluated when agents perform the same task.

    Example:

        Agent A -> tumor classification -> Healthy
        Agent B -> tumor classification -> HCC

    => conflict

    But:

        fibrosis -> stage 2
        cirrhosis -> stage 3

    are NOT considered a direct prediction conflict because they
    represent different tasks.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.20,
        uncertainty_threshold: float = 0.50,
    ):

        self.confidence_threshold = float(
            max(
                0.0,
                min(
                    1.0,
                    confidence_threshold
                )
            )
        )

        self.uncertainty_threshold = float(
            max(
                0.0,
                min(
                    1.0,
                    uncertainty_threshold
                )
            )
        )

    # -------------------------------------------------------------------------
    # DETECT
    # -------------------------------------------------------------------------

    def detect(self, messages):

        if not messages:
            return []

        valid_messages = []

        for message in messages:

            error = self._get(
                message,
                "error",
                None
            )

            status = self._get(
                message,
                "status",
                "success"
            )

            prediction = self._get(
                message,
                "prediction",
                None
            )

            if error is not None:
                continue

            if status not in (
                "success",
                "completed"
            ):
                continue

            if prediction is None:
                continue

            valid_messages.append(
                message
            )

        # -------------------------------------------------------------
        # GROUP BY TASK TYPE
        # -------------------------------------------------------------

        groups = {}

        for message in valid_messages:

            task_type = self._get(
                message,
                "task_type",
                None
            )

            if not task_type:

                details = self._get(
                    message,
                    "details",
                    {}
                ) or {}

                task_type = details.get(
                    "task_type",
                    "unknown"
                )

            groups.setdefault(
                task_type,
                []
            ).append(message)

        # -------------------------------------------------------------
        # PAIRWISE CONFLICT DETECTION
        # -------------------------------------------------------------

        conflicts = []

        for task_type, group in groups.items():

            if len(group) < 2:
                continue

            for i in range(len(group)):

                for j in range(
                    i + 1,
                    len(group)
                ):

                    a = group[i]
                    b = group[j]

                    prediction_a = self._get(
                        a,
                        "prediction"
                    )

                    prediction_b = self._get(
                        b,
                        "prediction"
                    )

                    if self._same_prediction(
                        prediction_a,
                        prediction_b
                    ):
                        continue

                    confidence_a = float(
                        self._get(
                            a,
                            "confidence",
                            0.0
                        )
                    )

                    confidence_b = float(
                        self._get(
                            b,
                            "confidence",
                            0.0
                        )
                    )

                    uncertainty_a = float(
                        self._get(
                            a,
                            "uncertainty",
                            1.0
                        )
                    )

                    uncertainty_b = float(
                        self._get(
                            b,
                            "uncertainty",
                            1.0
                        )
                    )

                    confidence_gap = abs(
                        confidence_a
                        - confidence_b
                    )

                    # -------------------------------------------------
                    # Conflict strength
                    # -------------------------------------------------

                    conflict_strength = (
                        (
                            confidence_a
                            +
                            confidence_b
                        ) / 2.0
                    )

                    conflict_strength *= (
                        1.0
                        -
                        (
                            uncertainty_a
                            +
                            uncertainty_b
                        ) / 2.0
                    )

                    conflicts.append({

                        "task_type":
                            task_type,

                        "agent_a":
                            self._get(
                                a,
                                "agent_id",
                                self._get(
                                    a,
                                    "agent",
                                    "unknown"
                                )
                            ),

                        "agent_b":
                            self._get(
                                b,
                                "agent_id",
                                self._get(
                                    b,
                                    "agent",
                                    "unknown"
                                )
                            ),

                        "prediction_a":
                            prediction_a,

                        "prediction_b":
                            prediction_b,

                        "confidence_a":
                            confidence_a,

                        "confidence_b":
                            confidence_b,

                        "uncertainty_a":
                            uncertainty_a,

                        "uncertainty_b":
                            uncertainty_b,

                        "confidence_gap":
                            confidence_gap,

                        "conflict_strength":
                            max(
                                0.0,
                                min(
                                    1.0,
                                    conflict_strength
                                )
                            ),

                        "requires_review":
                            (
                                confidence_gap
                                >=
                                self.confidence_threshold
                            ),
                    })

        return conflicts

    # -------------------------------------------------------------------------
    # SAME PREDICTION
    # -------------------------------------------------------------------------

    @staticmethod
    def _same_prediction(a, b):

        if a == b:
            return True

        try:

            import numpy as np

            return bool(
                np.array_equal(
                    np.asarray(a),
                    np.asarray(b)
                )
            )

        except Exception:

            return str(a) == str(b)

    # -------------------------------------------------------------------------
    # ACCESS
    # -------------------------------------------------------------------------

    @staticmethod
    def _get(
        obj,
        key,
        default=None
    ):

        if isinstance(obj, dict):

            return obj.get(
                key,
                default
            )

        return getattr(
            obj,
            key,
            default
        )
