class DecisionEngine:

    def __init__(
        self,
        confidence_threshold=0.70,
        uncertainty_threshold=0.40,
        agreement_threshold=0.60
    ):

        self.confidence_threshold = (
            confidence_threshold
        )

        self.uncertainty_threshold = (
            uncertainty_threshold
        )

        self.agreement_threshold = (
            agreement_threshold
        )

    def decide(
        self,
        agent_results,
        conflicts,
        fused_results,
        consensus=None
    ):

        if fused_results is None:

            return {

                "status":
                    "insufficient_evidence",

                "decision":
                    "additional_tests_required",

                "confidence":
                    0.0,

                "uncertainty":
                    1.0,

                "risk_score":
                    1.0,

                "request_additional_tests":
                    True,

                "reason":
                    "No reliable agent output."
            }

        confidence = float(
            fused_results.get(
                "confidence",
                0.0
            )
        )

        agreement = float(
            consensus.get(
                "agreement",
                0.0
            )
            if consensus
            else 0.0
        )

        uncertainty = 1.0 - confidence

        risk_score = (
            0.5 * uncertainty
            + 0.5 * (1.0 - agreement)
        )

        request_tests = (

            confidence
            < self.confidence_threshold

            or uncertainty
            > self.uncertainty_threshold

            or agreement
            < self.agreement_threshold
        )

        if request_tests:

            decision = (
                "additional_tests_required"
            )

            status = "uncertain"

        else:

            decision = (
                fused_results[
                    "prediction"
                ]
            )

            status = "completed"

        return {

            "status":
                status,

            "decision":
                decision,

            "confidence":
                confidence,

            "uncertainty":
                uncertainty,

            "risk_score":
                float(risk_score),

            "agreement":
                agreement,

            "conflicts":
                conflicts,

            "request_additional_tests":
                request_tests,

            "num_agents":
                len(agent_results)
        }
