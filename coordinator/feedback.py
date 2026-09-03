class FeedbackEngine:

    def __init__(
        self,
        trust_manager
    ):

        self.trust_manager = (
            trust_manager
        )

        self.feedback_history = []

    # =========================================================
    # UPDATE
    # =========================================================

    def update(
        self,
        agent_results,
        ground_truth
    ):

        updates = []

        for result in agent_results:

            prediction = result.get(
                "prediction"
            )

            agent_id = result.get(
                "agent_id",
                result.get(
                    "agent"
                )
            )

            if prediction is None:

                continue

            correct = (

                str(
                    prediction
                ).strip().lower()

                ==

                str(
                    ground_truth
                ).strip().lower()
            )

            new_performance = (

                self.trust_manager
                .update_from_outcome(

                    agent_id=agent_id,

                    correct=correct
                )
            )

            updates.append({

                "agent_id":
                    agent_id,

                "prediction":
                    prediction,

                "ground_truth":
                    ground_truth,

                "correct":
                    correct,

                "new_historical_performance":
                    new_performance
            })

        result = {

            "status":
                "updated",

            "ground_truth":
                ground_truth,

            "updates":
                updates
        }

        self.feedback_history.append(
            result
        )

        return result

    # =========================================================
    # HISTORY
    # =========================================================

    def get_history(self):

        return list(
            self.feedback_history
        )
