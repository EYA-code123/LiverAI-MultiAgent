class AdaptiveFusion:

    def fuse(self, results):

        valid_results = [
            r for r in results
            if r.status == "success"
            and r.prediction is not None
        ]

        if not valid_results:
            return None

        weighted_votes = {}

        for result in valid_results:

            trust = (
                result.trust
                if result.trust is not None
                else 0.5
            )

            confidence = (
                result.confidence
                if result.confidence is not None
                else 0.5
            )

            weight = trust * confidence

            prediction = str(
                result.prediction
            )

            weighted_votes[prediction] = (
                weighted_votes.get(
                    prediction,
                    0.0
                )
                + weight
            )

        final_prediction = max(
            weighted_votes,
            key=weighted_votes.get
        )

        total_weight = sum(
            weighted_votes.values()
        )

        final_confidence = (
            weighted_votes[final_prediction]
            / total_weight
            if total_weight > 0
            else 0.0
        )

        return {
            "prediction": final_prediction,
            "confidence": final_confidence,
            "weighted_votes": weighted_votes
        }
