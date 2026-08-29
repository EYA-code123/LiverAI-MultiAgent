class ConflictDetector:

    def detect(self, results):

        conflicts = []

        predictions = {}

        for result in results:

            if result.status != "success":
                continue

            predictions.setdefault(
                result.agent,
                result.prediction
            )

        unique_predictions = set(
            str(value)
            for value in predictions.values()
        )

        if len(unique_predictions) > 1:

            conflicts.append({
                "type": "prediction_disagreement",
                "agents": list(predictions.keys()),
                "predictions": predictions
            })

        return conflicts
