class ConflictDetector:

    def __init__(
        self,
        confidence_threshold=0.20
    ):

        self.confidence_threshold = (
            confidence_threshold
        )

    def detect(
        self,
        messages
    ):

        conflicts = []

        valid = [

            m for m in messages

            if m.error is None

            and m.prediction is not None

        ]

        groups = {}

        for message in valid:

            groups.setdefault(
                message.task_type,
                []
            ).append(message)

        for task_type, group in groups.items():

            for i in range(
                len(group)
            ):

                for j in range(
                    i + 1,
                    len(group)
                ):

                    a = group[i]
                    b = group[j]

                    if (
                        a.prediction
                        == b.prediction
                    ):
                        continue

                    confidence_gap = abs(
                        a.confidence
                        - b.confidence
                    )

                    severity = (
                        "high"
                        if confidence_gap >= 0.40
                        else "medium"
                        if confidence_gap >= 0.20
                        else "low"
                    )

                    conflicts.append({

                        "task_type":
                            task_type,

                        "agent_a":
                            a.agent_id,

                        "agent_b":
                            b.agent_id,

                        "prediction_a":
                            a.prediction,

                        "prediction_b":
                            b.prediction,

                        "confidence_a":
                            a.confidence,

                        "confidence_b":
                            b.confidence,

                        "confidence_gap":
                            confidence_gap,

                        "severity":
                            severity
                    })

        return conflicts

    def consensus(
        self,
        messages
    ):

        predictions = [

            m.prediction

            for m in messages

            if m.error is None

            and m.prediction is not None
        ]

        if not predictions:

            return {
                "agreement": 0.0,
                "consensus": None
            }

        counts = {}

        for prediction in predictions:

            key = str(prediction)

            counts[key] = (
                counts.get(
                    key,
                    0
                ) + 1
            )

        winner = max(
            counts,
            key=counts.get
        )

        agreement = (
            counts[winner]
            / len(predictions)
        )

        return {

            "agreement":
                float(agreement),

            "consensus":
                winner,

            "counts":
                counts
        }
