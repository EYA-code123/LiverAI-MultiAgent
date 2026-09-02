"""
Feedback Intelligence
======================

Updates agent historical performance when a verified
ground-truth outcome becomes available.

IMPORTANT:
Never call this with a guessed clinical outcome.
"""


class FeedbackIntelligence:

    def __init__(
        self,
        trust_manager
    ):

        self.trust_manager = (
            trust_manager
        )

        self.history = []

    def update(
        self,
        messages,
        ground_truths
    ):

        updates = []

        for message in messages:

            if message.agent_id not in ground_truths:
                continue

            truth = ground_truths[
                message.agent_id
            ]

            prediction = message.prediction

            if prediction is None:
                continue

            try:

                correct = (
                    str(prediction)
                    ==
                    str(truth)
                )

            except Exception:

                correct = False

            new_performance = (
                self.trust_manager
                .update_historical_performance(
                    message.agent_id,
                    int(correct)
                )
            )

            updates.append({
                "agent_id":
                    message.agent_id,

                "prediction":
                    prediction,

                "ground_truth":
                    truth,

                "correct":
                    correct,

                "new_historical_performance":
                    new_performance
            })

        self.history.extend(
            updates
        )

        return {
            "updates": updates,
            "num_updates": len(updates)
        }
