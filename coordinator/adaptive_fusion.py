import numpy as np


class AdaptiveFusion:

    def __init__(self):

        self.temperature = 1.0

    def _softmax(self, values):

        values = np.asarray(
            values,
            dtype=float
        )

        values = values / max(
            self.temperature,
            1e-6
        )

        values = (
            values
            - np.max(values)
        )

        exp_values = np.exp(values)

        return (
            exp_values
            /
            (
                np.sum(exp_values)
                + 1e-8
            )
        )

    def fuse(self, results):

        valid = [

            r for r in results

            if r.status == "success"

            and r.prediction is not None
        ]

        if not valid:

            return {
                "prediction": None,
                "confidence": 0.0,
                "weights": {},
                "support": 0
            }

        scores = []

        for r in valid:

            score = (

                r.trust

                * r.confidence

                * (1.0 - r.uncertainty)

                * r.quality
            )

            scores.append(score)

        weights = self._softmax(
            scores
        )

        votes = {}

        weight_map = {}

        for r, weight in zip(
            valid,
            weights
        ):

            prediction = str(
                r.prediction
            )

            weight = float(weight)

            weight_map[
                r.agent_id
            ] = weight

            votes[
                prediction
            ] = (
                votes.get(
                    prediction,
                    0.0
                )
                + weight
            )

        final_prediction = max(
            votes,
            key=votes.get
        )

        final_confidence = float(
            votes[
                final_prediction
            ]
        )

        return {

            "prediction":
                final_prediction,

            "confidence":
                final_confidence,

            "weights":
                weight_map,

            "weighted_votes":
                votes,

            "support":
                len(valid)
        }
