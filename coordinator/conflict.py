class ConflictDetector:

    def __init__(
        self,
        prediction_threshold=0.5,
        confidence_threshold=0.20
    ):

        self.prediction_threshold = prediction_threshold
        self.confidence_threshold = confidence_threshold

    def detect(self, messages):

        conflicts = []

        valid_messages = [
            m for m in messages
            if m.error is None
        ]

        for i in range(len(valid_messages)):

            for j in range(i + 1, len(valid_messages)):

                a = valid_messages[i]
                b = valid_messages[j]

                # Les agents spécialisés peuvent avoir
                # des espaces de prédiction différents.
                # On compare donc uniquement les agents
                # ayant le même type de tâche.

                task_a = a.details.get(
                    "task_type"
                )

                task_b = b.details.get(
                    "task_type"
                )

                if task_a != task_b:
                    continue

                if a.prediction != b.prediction:

                    confidence_gap = abs(
                        a.confidence -
                        b.confidence
                    )

                    if confidence_gap >= self.confidence_threshold:

                        conflicts.append({
                            "agent_a": a.agent_id,
                            "agent_b": b.agent_id,
                            "prediction_a": a.prediction,
                            "prediction_b": b.prediction,
                            "confidence_a": a.confidence,
                            "confidence_b": b.confidence,
                            "confidence_gap": confidence_gap
                        })

        return conflicts
