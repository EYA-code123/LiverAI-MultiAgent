import time
import traceback


class AgentAdapter:

    def __init__(
        self,
        agent_id,
        agent,
        task_type,
        modality="unknown"
    ):

        self.agent_id = agent_id
        self.agent = agent
        self.task_type = task_type
        self.modality = modality

    # =========================================================
    # RUN
    # =========================================================

    def predict(
        self,
        patient_id,
        data
    ):

        start = time.perf_counter()

        try:

            if self.agent is None:

                return {

                    "patient_id":
                        patient_id,

                    "agent_id":
                        self.agent_id,

                    "agent":
                        self.agent_id,

                    "task_type":
                        self.task_type,

                    "modality":
                        self.modality,

                    "prediction":
                        None,

                    "status":
                        "unavailable",

                    "error":
                        "Agent is not loaded."
                }

            # -------------------------------------------------
            # Try predict()
            # -------------------------------------------------

            if hasattr(
                self.agent,
                "predict"
            ):

                output = (
                    self.agent.predict(
                        data
                    )
                )

            # -------------------------------------------------
            # Try run()
            # -------------------------------------------------

            elif hasattr(
                self.agent,
                "run"
            ):

                output = (
                    self.agent.run(
                        data
                    )
                )

            else:

                raise AttributeError(
                    "Agent must expose "
                    "predict() or run()."
                )

            latency_ms = (

                time.perf_counter()
                - start
            ) * 1000.0

            if not isinstance(
                output,
                dict
            ):

                output = {

                    "prediction":
                        output
                }

            output = dict(
                output
            )

            output.setdefault(
                "patient_id",
                patient_id
            )

            output.setdefault(
                "agent_id",
                self.agent_id
            )

            output.setdefault(
                "agent",
                self.agent_id
            )

            output.setdefault(
                "task_type",
                self.task_type
            )

            output.setdefault(
                "modality",
                self.modality
            )

            output.setdefault(
                "status",
                "success"
            )

            output[
                "latency_ms"
            ] = latency_ms

            # -------------------------------------------------
            # Normalize
            # -------------------------------------------------

            return self.normalize(
                output
            )

        except Exception as exc:

            latency_ms = (

                time.perf_counter()
                - start
            ) * 1000.0

            return {

                "patient_id":
                    patient_id,

                "agent_id":
                    self.agent_id,

                "agent":
                    self.agent_id,

                "task_type":
                    self.task_type,

                "modality":
                    self.modality,

                "prediction":
                    None,

                "probability":
                    None,

                "class_probabilities":
                    {},

                "confidence":
                    0.0,

                "uncertainty":
                    1.0,

                "quality":
                    0.0,

                "missing_data_ratio":
                    1.0,

                "status":
                    "error",

                "error":
                    str(exc),

                "traceback":
                    traceback.format_exc(),

                "latency_ms":
                    latency_ms
            }

    # =========================================================
    # NORMALIZE
    # =========================================================

    def normalize(
        self,
        result
    ):

        result.setdefault(
            "prediction",
            None
        )

        result.setdefault(
            "probability",
            None
        )

        result.setdefault(
            "class_probabilities",
            {}
        )

        result.setdefault(
            "confidence",
            0.0
        )

        result.setdefault(
            "uncertainty",
            1.0
        )

        result.setdefault(
            "quality",
            1.0
        )

        result.setdefault(
            "missing_data_ratio",
            0.0
        )

        result.setdefault(
            "agreement",
            0.5
        )

        result.setdefault(
            "stability",
            0.5
        )

        result.setdefault(
            "utility",
            0.5
        )

        result.setdefault(
            "feature_importance",
            {}
        )

        result.setdefault(
            "explanation",
            None
        )

        return result
