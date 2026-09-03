from itertools import combinations


class ConflictDetector:

    def __init__(
        self,
        confidence_gap_threshold=0.20
    ):

        self.confidence_gap_threshold = float(
            confidence_gap_threshold
        )

    # =========================================================
    # MAIN
    # =========================================================

    def detect(
        self,
        results
    ):

        valid = [

            result

            for result in results

            if isinstance(
                result,
                dict
            )

            and result.get(
                "status",
                "success"
            )
            in (
                "success",
                "completed"
            )

            and result.get(
                "prediction"
            ) is not None
        ]

        # -----------------------------------------------------
        # GROUP BY TASK
        # -----------------------------------------------------

        groups = {}

        for result in valid:

            task = result.get(
                "task_type",
                "unknown"
            )

            groups.setdefault(
                task,
                []
            ).append(
                result
            )

        conflicts = []

        for task, agents in (
            groups.items()
        ):

            if len(agents) < 2:
                continue

            for a, b in combinations(
                agents,
                2
            ):

                prediction_a = a.get(
                    "prediction"
                )

                prediction_b = b.get(
                    "prediction"
                )

                if prediction_a == prediction_b:

                    continue

                confidence_a = float(
                    a.get(
                        "confidence",
                        0.0
                    )
                )

                confidence_b = float(
                    b.get(
                        "confidence",
                        0.0
                    )
                )

                gap = abs(
                    confidence_a
                    - confidence_b
                )

                # -------------------------------------------------
                # Conflict strength
                # -------------------------------------------------

                conflict_strength = min(

                    1.0,

                    0.50
                    + 0.50 * gap
                )

                conflicts.append({

                    "task_type":
                        task,

                    "agent_1":
                        a.get(
                            "agent_id",
                            a.get(
                                "agent"
                            )
                        ),

                    "prediction_1":
                        prediction_a,

                    "confidence_1":
                        confidence_a,

                    "trust_1":
                        float(
                            a.get(
                                "trust",
                                0.0
                            )
                        ),

                    "agent_2":
                        b.get(
                            "agent_id",
                            b.get(
                                "agent"
                            )
                        ),

                    "prediction_2":
                        prediction_b,

                    "confidence_2":
                        confidence_b,

                    "trust_2":
                        float(
                            b.get(
                                "trust",
                                0.0
                            )
                        ),

                    "confidence_gap":
                        gap,

                    "conflict_strength":
                        conflict_strength
                })

        return conflicts

    # =========================================================
    # AGREEMENT
    # =========================================================

    def compute_agreement(
        self,
        results
    ):

        predictions = [

            r.get(
                "prediction"
            )

            for r in results

            if r.get(
                "prediction"
            ) is not None
        ]

        if len(predictions) < 2:

            return 1.0

        majority = max(

            set(predictions),

            key=predictions.count
        )

        return (

            predictions.count(
                majority
            )
            /
            len(predictions)
        )
