# =============================================================================
# LiverAI-MultiAgent
# ADAPTIVE EVIDENCE FUSION
# =============================================================================

from collections import defaultdict


class AdaptiveFusion:
    """
    Adaptive evidence fusion for heterogeneous medical agents.

    IMPORTANT:
    We do NOT perform a global majority vote between heterogeneous
    tasks such as:

        cirrhosis
        fibrosis
        fatty liver
        tumor
        segmentation

    Instead, each agent keeps its own task prediction and receives
    an adaptive contribution weight.

    If several agents perform the SAME task, their probabilities
    can also be fused.
    """

    def __init__(
        self,
        minimum_trust: float = 0.05,
    ):

        self.minimum_trust = float(
            max(
                0.0,
                min(
                    1.0,
                    minimum_trust
                )
            )
        )

    # -------------------------------------------------------------------------
    # MAIN FUSION
    # -------------------------------------------------------------------------

    def fuse(self, results):

        if not results:
            return {
                "status": "no_results",
                "evidence": [],
                "task_groups": {},
                "weights": {},
            }

        valid_results = []

        for result in results:

            if result is None:
                continue

            status = self._get(
                result,
                "status",
                "success"
            )

            prediction = self._get(
                result,
                "prediction",
                None
            )

            if status not in (
                "success",
                "completed"
            ):
                continue

            if prediction is None:
                continue

            valid_results.append(
                result
            )

        if not valid_results:

            return {
                "status": "no_valid_results",
                "evidence": [],
                "task_groups": {},
                "weights": {},
            }

        # -------------------------------------------------------------
        # CALCULATE RAW WEIGHTS
        # -------------------------------------------------------------

        raw_weights = {}

        for result in valid_results:

            agent_id = self._get(
                result,
                "agent_id",
                self._get(
                    result,
                    "agent",
                    "unknown"
                )
            )

            trust = self._get(
                result,
                "trust",
                None
            )

            if trust is None:
                trust = 0.5

            confidence = self._get(
                result,
                "confidence",
                0.0
            )

            quality = self._get(
                result,
                "quality",
                0.0
            )

            uncertainty = self._get(
                result,
                "uncertainty",
                1.0
            )

            # Patient-specific adaptive weight
            weight = (
                float(trust)
                *
                float(confidence)
                *
                float(quality)
                *
                (1.0 - float(uncertainty))
            )

            weight = max(
                self.minimum_trust,
                weight
            )

            raw_weights[
                str(agent_id)
            ] = weight

        # -------------------------------------------------------------
        # NORMALIZE WEIGHTS
        # -------------------------------------------------------------

        total_weight = sum(
            raw_weights.values()
        )

        if total_weight <= 0:

            normalized_weights = {
                agent_id: 0.0
                for agent_id in raw_weights
            }

        else:

            normalized_weights = {
                agent_id:
                    weight / total_weight
                for agent_id, weight
                in raw_weights.items()
            }

        # -------------------------------------------------------------
        # BUILD EVIDENCE
        # -------------------------------------------------------------

        evidence = []

        for result in valid_results:

            agent_id = str(
                self._get(
                    result,
                    "agent_id",
                    self._get(
                        result,
                        "agent",
                        "unknown"
                    )
                )
            )

            task_type = self._get(
                result,
                "task_type",
                None
            )

            if not task_type:

                details = self._get(
                    result,
                    "details",
                    {}
                ) or {}

                task_type = details.get(
                    "task_type",
                    "unknown"
                )

            evidence.append({

                "agent_id": agent_id,

                "task_type": task_type,

                "prediction": self._get(
                    result,
                    "prediction"
                ),

                "probability": self._get(
                    result,
                    "probability"
                ),

                "confidence": self._get(
                    result,
                    "confidence",
                    0.0
                ),

                "uncertainty": self._get(
                    result,
                    "uncertainty",
                    1.0
                ),

                "quality": self._get(
                    result,
                    "quality",
                    0.0
                ),

                "trust": self._get(
                    result,
                    "trust",
                    0.0
                ),

                "adaptive_weight":
                    normalized_weights.get(
                        agent_id,
                        0.0
                    ),

                "explanation":
                    self._get(
                        result,
                        "explanation",
                        None
                    ),
            })

        # -------------------------------------------------------------
        # GROUP BY TASK
        # -------------------------------------------------------------

        task_groups = defaultdict(list)

        for item in evidence:

            task_groups[
                item["task_type"]
            ].append(item)

        task_groups = dict(
            task_groups
        )

        # -------------------------------------------------------------
        # OPTIONAL SAME-TASK PROBABILITY FUSION
        # -------------------------------------------------------------

        fused_tasks = {}

        for task_type, items in task_groups.items():

            if len(items) <= 1:
                continue

            fused = self._fuse_same_task(
                items
            )

            if fused is not None:
                fused_tasks[
                    task_type
                ] = fused

        return {
            "status": "success",

            "evidence": evidence,

            "weights": normalized_weights,

            "task_groups": task_groups,

            "same_task_fusion":
                fused_tasks,

            "num_valid_agents":
                len(valid_results),
        }

    # -------------------------------------------------------------------------
    # SAME TASK FUSION
    # -------------------------------------------------------------------------

    def _fuse_same_task(self, items):

        probability_vectors = []

        for item in items:

            probability = item.get(
                "probability"
            )

            if not isinstance(
                probability,
                (list, tuple)
            ):
                continue

            try:

                vector = [
                    float(x)
                    for x in probability
                ]

            except (
                TypeError,
                ValueError
            ):

                continue

            if not vector:
                continue

            probability_vectors.append(
                (
                    vector,
                    item[
                        "adaptive_weight"
                    ]
                )
            )

        if not probability_vectors:
            return None

        dimension = len(
            probability_vectors[0][0]
        )

        if any(
            len(vector) != dimension
            for vector, _ in probability_vectors
        ):
            return None

        fused = [
            0.0
            for _ in range(dimension)
        ]

        total_weight = 0.0

        for vector, weight in probability_vectors:

            total_weight += weight

            for i in range(dimension):

                fused[i] += (
                    vector[i]
                    * weight
                )

        if total_weight > 0:

            fused = [
                value / total_weight
                for value in fused
            ]

        best_index = max(
            range(len(fused)),
            key=lambda i: fused[i]
        )

        return {
            "probabilities": fused,
            "predicted_class_index":
                best_index,

            "confidence":
                fused[best_index],

            "num_agents":
                len(probability_vectors),
        }

    # -------------------------------------------------------------------------
    # DICTIONARY / OBJECT ACCESS
    # -------------------------------------------------------------------------

    @staticmethod
    def _get(
        obj,
        key,
        default=None
    ):

        if isinstance(obj, dict):

            return obj.get(
                key,
                default
            )

        return getattr(
            obj,
            key,
            default
        )
