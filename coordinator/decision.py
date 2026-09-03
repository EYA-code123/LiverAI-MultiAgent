class DecisionEngine:

    def __init__(
        self,
        high_confidence=0.80,
        moderate_confidence=0.55,
        high_trust=0.70,
        high_conflict=0.50,
        minimum_coverage=0.50,
        minimum_quality=0.50
    ):

        self.high_confidence = float(
            high_confidence
        )

        self.moderate_confidence = float(
            moderate_confidence
        )

        self.high_trust = float(
            high_trust
        )

        self.high_conflict = float(
            high_conflict
        )

        self.minimum_coverage = float(
            minimum_coverage
        )

        self.minimum_quality = float(
            minimum_quality
        )

    # =========================================================
    # DECISION
    # =========================================================

    def decide(
        self,
        results,
        conflicts=None,
        reasoning=None
    ):

        conflicts = conflicts or {}

        reasoning = reasoning or {}

        valid = [

            r

            for r in results

            if r.get(
                "prediction"
            ) is not None

            and r.get(
                "status",
                "success"
            )
            in (
                "success",
                "completed"
            )
        ]

        total_agents = len(
            results
        )

        valid_agents = len(
            valid
        )

        coverage = (

            valid_agents
            /
            total_agents

            if total_agents > 0

            else 0.0
        )

        # -----------------------------------------------------
        # AVERAGES
        # -----------------------------------------------------

        if valid:

            mean_confidence = sum(

                float(
                    r.get(
                        "confidence",
                        0.0
                    )
                )

                for r in valid

            ) / len(valid)

            mean_trust = sum(

                float(
                    r.get(
                        "trust",
                        0.0
                    )
                )

                for r in valid

            ) / len(valid)

            mean_quality = sum(

                float(
                    r.get(
                        "quality",
                        0.0
                    )
                )

                for r in valid

            ) / len(valid)

        else:

            mean_confidence = 0.0
            mean_trust = 0.0
            mean_quality = 0.0

        # -----------------------------------------------------
        # CONFLICT
        # -----------------------------------------------------

        if isinstance(
            conflicts,
            dict
        ):

            conflict_values = []

            for value in (
                conflicts.values()
            ):

                if isinstance(
                    value,
                    dict
                ):

                    conflict_values.append(

                        float(
                            value.get(
                                "conflict_strength",
                                0.0
                            )
                        )
                    )

            conflict_score = (

                sum(
                    conflict_values
                )
                /
                len(
                    conflict_values
                )

                if conflict_values

                else 0.0
            )

        else:

            conflict_score = (

                sum(

                    float(
                        c.get(
                            "conflict_strength",
                            0.0
                        )
                    )

                    for c in conflicts
                )
                /
                len(conflicts)

                if conflicts

                else 0.0
            )

        # -----------------------------------------------------
        # PREDICTION
        # -----------------------------------------------------

        prediction = reasoning.get(
            "prediction"
        )

        if prediction is None and valid:

            best = max(

                valid,

                key=lambda r:
                    float(
                        r.get(
                            "trust",
                            0.0
                        )
                    )
                    *
                    float(
                        r.get(
                            "confidence",
                            0.0
                        )
                    )
            )

            prediction = best.get(
                "prediction"
            )

        # -----------------------------------------------------
        # DECISION POLICY
        # -----------------------------------------------------

        insufficient_data = (

            coverage
            < self.minimum_coverage

            or mean_quality
            < self.minimum_quality
        )

        unsafe_conflict = (
            conflict_score
            >= self.high_conflict
        )

        if (

            not valid

            or insufficient_data

            or unsafe_conflict

            or mean_confidence
            < self.moderate_confidence
        ):

            decision_level = "UNCERTAIN"

        elif (

            mean_confidence
            >= self.high_confidence

            and mean_trust
            >= self.high_trust

            and conflict_score
            < 0.30

        ):

            decision_level = "HIGH"

        else:

            decision_level = "MODERATE"

        # -----------------------------------------------------
        # RISK
        # -----------------------------------------------------

        risk_score = (

            0.40 * (
                1.0
                - mean_confidence
            )

            +

            0.30 * (
                1.0
                - mean_trust
            )

            +

            0.20 * conflict_score

            +

            0.10 * (
                1.0
                - mean_quality
            )
        )

        risk_score = max(
            0.0,
            min(
                1.0,
                risk_score
            )
        )

        return {

            "status":
                "completed",

            "prediction":
                prediction,

            "decision_level":
                decision_level,

            "confidence":
                float(
                    mean_confidence
                ),

            "uncertainty":
                float(
                    1.0 - mean_confidence
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
                decision_level
                == "UNCERTAIN",

            "num_agents":
                total_agents,

            "num_valid_agents":
                valid_agents
        }
