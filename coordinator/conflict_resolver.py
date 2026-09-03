class ConflictResolver:

    def __init__(
        self,
        consensus_threshold=0.60
    ):

        self.consensus_threshold = float(
            consensus_threshold
        )

    # =========================================================
    # RESOLVE
    # =========================================================

    def resolve(
        self,
        task_type,
        results,
        conflicts=None
    ):

        conflicts = conflicts or []

        valid = [

            r

            for r in results

            if r.get(
                "task_type"
            ) == task_type

            and r.get(
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

        if not valid:

            return {

                "status":
                    "unresolved",

                "consensus":
                    False,

                "prediction":
                    None,

                "consensus_strength":
                    0.0,

                "reason":
                    "No valid evidence."
            }

        # -----------------------------------------------------
        # SCORE EACH PREDICTION
        # -----------------------------------------------------

        scores = {}

        for result in valid:

            prediction = result[
                "prediction"
            ]

            trust = float(
                result.get(
                    "trust",
                    0.0
                )
            )

            confidence = float(
                result.get(
                    "confidence",
                    0.0
                )
            )

            quality = float(
                result.get(
                    "quality",
                    0.0
                )
            )

            stability = float(
                result.get(
                    "stability",
                    0.5
                )
            )

            score = (

                trust
                * confidence
                * quality
                * (
                    0.5
                    + 0.5 * stability
                )
            )

            scores[
                prediction
            ] = (

                scores.get(
                    prediction,
                    0.0
                )
                +
                score
            )

        total = sum(
            scores.values()
        )

        if total <= 0:

            return {

                "status":
                    "unresolved",

                "consensus":
                    False,

                "prediction":
                    None,

                "consensus_strength":
                    0.0
            }

        winner = max(
            scores,
            key=scores.get
        )

        consensus_strength = (

            scores[winner]
            /
            total
        )

        # -----------------------------------------------------
        # REASON
        # -----------------------------------------------------

        if not conflicts:

            reason = (
                "No disagreement detected."
            )

        elif consensus_strength >= (
            self.consensus_threshold
        ):

            reason = (
                "Conflict resolved using "
                "trust, confidence, quality "
                "and stability."
            )

        else:

            reason = (
                "Evidence remains insufficient "
                "for a reliable consensus."
            )

        return {

            "status":
                "resolved",

            "consensus":
                consensus_strength
                >= self.consensus_threshold,

            "prediction":
                winner,

            "consensus_strength":
                float(
                    consensus_strength
                ),

            "scores":
                scores,

            "reason":
                reason
        } 
