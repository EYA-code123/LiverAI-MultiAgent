class DecisionEngine:

    def __init__(
        self,
        high_confidence=0.80,
        moderate_confidence=0.55,
        high_trust=0.70,
        high_conflict=0.50,
        minimum_coverage=0.50,
        minimum_quality=0.50,
    ):
        self.high_confidence = high_confidence
        self.moderate_confidence = moderate_confidence
        self.high_trust = high_trust
        self.high_conflict = high_conflict
        self.minimum_coverage = minimum_coverage
        self.minimum_quality = minimum_quality

    # ============================================================
    # VALIDATE RESULT
    # ============================================================

    def _is_valid_result(self, result):

        if not isinstance(result, dict):
            return False

        if result.get("status") not in (
            "success",
            "completed",
        ):
            return False

        task_type = result.get(
            "task_type",
            result.get("task", "")
        )

        # --------------------------------------------------------
        # SEGMENTATION
        # --------------------------------------------------------

        if task_type == "liver_segmentation":

            details = result.get(
                "details",
                {}
            )

            if not isinstance(details, dict):
                return False

            has_mask = (
                details.get("liver_mask") is not None
            )

            has_probability_map = (
                details.get("probability_map") is not None
            )

            return (
                has_mask
                or
                has_probability_map
            )

        # --------------------------------------------------------
        # OTHER TASKS
        # --------------------------------------------------------

        return (
            result.get("prediction") is not None
        )

    # ============================================================
    # DECIDE
    # ============================================================

    def decide(
        self,
        results,
        conflicts=None,
        reasoning=None,
    ):

        conflicts = conflicts or []
        reasoning = reasoning or {}

        # ========================================================
        # VALID RESULTS
        # ========================================================

        valid_results = []

        for result in results:

            if self._is_valid_result(result):
                valid_results.append(result)

        total_agents = len(results)

        valid_agents = len(
            valid_results
        )

        coverage = (
            valid_agents / total_agents
            if total_agents > 0
            else 0.0
        )

        # ========================================================
        # MEAN CONFIDENCE / TRUST / QUALITY
        # ========================================================

        if valid_results:

            mean_confidence = sum(
                float(
                    r.get(
                        "confidence",
                        0.0
                    )
                )
                for r in valid_results
            ) / valid_agents

            mean_trust = sum(
                float(
                    r.get(
                        "trust",
                        0.0
                    )
                )
                for r in valid_results
            ) / valid_agents

            mean_quality = sum(
                float(
                    r.get(
                        "quality",
                        0.0
                    )
                )
                for r in valid_results
            ) / valid_agents

        else:

            mean_confidence = 0.0
            mean_trust = 0.0
            mean_quality = 0.0

        # ========================================================
        # CONFLICT SCORE
        # ========================================================

        conflict_values = []

        for conflict in conflicts:

            if not isinstance(
                conflict,
                dict
            ):
                continue

            value = conflict.get(
                "conflict_strength"
            )

            if value is None:

                gap = conflict.get(
                    "confidence_gap",
                    0.0
                )

                value = min(
                    1.0,
                    float(gap)
                )

            conflict_values.append(
                float(value)
            )

        conflict_score = (
            sum(conflict_values)
            /
            len(conflict_values)
            if conflict_values
            else 0.0
        )

        # ========================================================
        # PREDICTION
        # ========================================================

        prediction = None

        if isinstance(
            reasoning,
            dict
        ):

            prediction = reasoning.get(
                "prediction"
            )

        # Segmentation must NOT become a class prediction.
        prediction_candidates = [
            r
            for r in valid_results
            if r.get(
                "task_type",
                r.get("task", "")
            ) != "liver_segmentation"
            and
            r.get("prediction") is not None
        ]

        if (
            prediction is None
            and prediction_candidates
        ):

            best = max(
                prediction_candidates,
                key=lambda x:
                    float(
                        x.get(
                            "trust",
                            0.0
                        )
                    )
                    *
                    float(
                        x.get(
                            "confidence",
                            0.0
                        )
                    )
            )

            prediction = best.get(
                "prediction"
            )

        # ========================================================
        # DECISION LEVEL
        # ========================================================

        insufficient_data = (
            coverage <
            self.minimum_coverage
            or
            mean_quality <
            self.minimum_quality
        )

        unsafe_conflict = (
            conflict_score >=
            self.high_conflict
        )

        if (
            not valid_results
            or insufficient_data
            or unsafe_conflict
            or mean_confidence <
               self.moderate_confidence
        ):

            decision_level = "UNCERTAIN"

        elif (
            mean_confidence >=
            self.high_confidence
            and
            mean_trust >=
            self.high_trust
            and
            conflict_score < 0.30
        ):

            decision_level = "HIGH"

        else:

            decision_level = "MODERATE"

        # ========================================================
        # RISK
        # ========================================================

        risk_score = (

            0.40 *
            (1.0 - mean_confidence)

            +

            0.30 *
            (1.0 - mean_trust)

            +

            0.20 *
            conflict_score

            +

            0.10 *
            (1.0 - mean_quality)
        )

        risk_score = max(
            0.0,
            min(
                1.0,
                risk_score
            )
        )

        # ========================================================
        # FINAL RESULT
        # ========================================================

        return {

            "status":
                "completed",

            "decision":
                decision_level,

            "prediction":
                prediction,

            "confidence":
                float(
                    mean_confidence
                ),

            "uncertainty":
                float(
                    1.0 -
                    mean_confidence
                ),

            "trust":
                float(
                    mean_trust
                ),

            "quality":
                float(
                    mean_quality
                ),

            "coverage":
                float(
                    coverage
                ),

            "conflict_score":
                float(
                    conflict_score
                ),

            "risk_score":
                float(
                    risk_score
                ),

            "request_additional_tests":
                decision_level ==
                "UNCERTAIN",

            "num_agents":
                total_agents,

            "num_valid_agents":
                valid_agents,

            "explanation":
                (
                    f"{valid_agents}/"
                    f"{total_agents} agents "
                    f"provided valid evidence. "
                    f"Mean confidence="
                    f"{mean_confidence:.3f}, "
                    f"mean trust="
                    f"{mean_trust:.3f}, "
                    f"conflict="
                    f"{conflict_score:.3f}."
                ),
        }
