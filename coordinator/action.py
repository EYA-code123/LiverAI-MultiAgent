class ActionEngine:

    def generate(
        self,
        decision
    ):

        level = decision.get(
            "decision_level",
            "UNCERTAIN"
        )

        prediction = decision.get(
            "prediction"
        )

        confidence = float(
            decision.get(
                "confidence",
                0.0
            )
        )

        risk = float(
            decision.get(
                "risk_score",
                1.0
            )
        )

        request_tests = decision.get(
            "request_additional_tests",
            False
        )

        # =====================================================
        # UNCERTAIN
        # =====================================================

        if request_tests:

            return {

                "status":
                    "cautious",

                "actions": [

                    "Additional clinical assessment recommended.",

                    "Consider additional imaging or laboratory data.",

                    "Specialist review recommended.",

                    "Do not use this automated output as a standalone diagnosis."
                ],

                "referral":
                    True,

                "follow_up":
                    True,

                "additional_tests":
                    True
            }

        # =====================================================
        # HIGH
        # =====================================================

        if level == "HIGH":

            return {

                "status":
                    "high_confidence",

                "actions": [

                    f"Finding requiring clinical validation: {prediction}.",

                    "Review the supporting evidence.",

                    "Consider specialist confirmation before intervention."
                ],

                "referral":
                    True,

                "follow_up":
                    True,

                "additional_tests":
                    False,

                "risk_score":
                    risk,

                "confidence":
                    confidence
            }

        # =====================================================
        # MODERATE
        # =====================================================

        if level == "MODERATE":

            return {

                "status":
                    "moderate_confidence",

                "actions": [

                    f"Preliminary finding: {prediction}.",

                    "Perform clinical review.",

                    "Consider additional evidence if clinically indicated."
                ],

                "referral":
                    True,

                "follow_up":
                    True,

                "additional_tests":
                    True,

                "risk_score":
                    risk,

                "confidence":
                    confidence
            }

        # =====================================================
        # FALLBACK
        # =====================================================

        return {

            "status":
                "uncertain",

            "actions": [

                "Insufficient evidence.",

                "Additional assessment required."
            ],

            "referral":
                True,

            "follow_up":
                True,

            "additional_tests":
                True
        }
