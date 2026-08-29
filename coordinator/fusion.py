import numpy as np


class FusionEngine:

    def fuse_classification(
        self,
        messages,
        trust_scores
    ):

        valid = [
            m for m in messages
            if m.error is None
            and m.probability is not None
        ]

        if not valid:
            return {
                "prediction": None,
                "probabilities": None,
                "confidence": 0.0,
                "support": 0
            }

        weighted_probabilities = []
        weights = []

        for message in valid:

            probabilities = np.asarray(
                message.probability,
                dtype=float
            )

            weight = float(
                trust_scores.get(
                    message.agent_id,
                    0.0
                )
            )

            if weight <= 0:
                continue

            weighted_probabilities.append(
                probabilities * weight
            )

            weights.append(weight)

        if not weighted_probabilities:

            return {
                "prediction": None,
                "probabilities": None,
                "confidence": 0.0,
                "support": 0
            }

        weighted_probabilities = np.asarray(
            weighted_probabilities
        )

        weights = np.asarray(weights)

        fused = (
            weighted_probabilities.sum(axis=0)
            / weights.sum()
        )

        prediction = int(
            np.argmax(fused)
        )

        confidence = float(
            np.max(fused)
        )

        return {
            "prediction": prediction,
            "probabilities": fused.tolist(),
            "confidence": confidence,
            "support": len(valid)
        } 
