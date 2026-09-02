class FeedbackEngine:

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
        agent_results,
        ground_truth
    ):

        feedback = []

        for result in agent_results:

            if result.prediction is None:
                continue

            correct = (
                str(result.prediction)
                == str(ground_truth)
            )

            new_performance = (
                self.trust_manager
                .update_from_feedback(
                    result.agent_id,
                    correct
                )
            )

            feedback.append({

                "agent_id":
                    result.agent_id,

                "prediction":
                    result.prediction,

                "ground_truth":
                    ground_truth,

                "correct":
                    correct,

                "updated_performance":
                    new_performance
            })

        self.history.append(
            feedback
        )

        return feedback
