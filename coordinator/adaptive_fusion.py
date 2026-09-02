# =============================================================================
# LiverAI-MultiAgent
# ADAPTIVE EVIDENCE FUSION
# =============================================================================

import numpy as np


class AdaptiveFusion:

    def __init__(self):

        self.minimum_weight = 1e-8

    # -------------------------------------------------------------------------
    # COMPUTE AGENT WEIGHT
    # -------------------------------------------------------------------------

    def _compute_weight(
        self,
        result
    ):

        trust = float(
            getattr(
                result,
                "trust",
                0.5
            )
        )

        confidence = float(
            getattr(
                result,
                "confidence",
                0.0
            )
        )

        quality = float(
            getattr(
                result,
                "quality",
                0.0
            )
        )

        uncertainty = float(
            getattr(
                result,
                "uncertainty",
                1.0
            )
        )

        weight = (
            trust
            * confidence
            * quality
            * (1.0 - uncertainty)
        )

        return max(
            self.minimum_weight,
            weight
        )

    # -------------------------------------------------------------------------
    # ADAPTIVE EVIDENCE FUSION
    # -------------------------------------------------------------------------

    def fuse(
        self,
        results
    ):

        valid_results = [
            r for r in results
            if getattr(r, "success", False)
        ]

        if not valid_results:

            return {
                "status": "no_valid_agents",
                "weights": {},
                "evidence": [],
                "total_weight": 0.0,
                "confidence": 0.0
            }

        raw_weights = {}

        for result in valid_results:

            raw_weights[
                result.agent_id
            ] = self._compute_weight(
                result
            )

        total_weight = sum(
            raw_weights.values()
        )

        normalized_weights = {
            agent_id:
                weight / total_weight
            for agent_id, weight
            in raw_weights.items()
        }

        evidence = []

        for result in valid_results:

            agent_id = result.agent_id

            evidence.append({

                "agent_id":
                    agent_id,

                "task_type":
                    getattr(
                        result,
                        "task_type",
                        "unknown"
                    ),

                "prediction":
                    result.prediction,

                "confidence":
                    float(
                        result.confidence
                    ),

                "uncertainty":
                    float(
                        result.uncertainty
                    ),

                "quality":
                    float(
                        result.quality
                    ),

                "trust":
                    float(
                        result.trust
                    ),

                "weight":
                    float(
                        normalized_weights[
                            agent_id
                        ]
                    ),

                "contribution":
                    float(
                        normalized_weights[
                            agent_id
                        ]
                        * result.confidence
                    )
            })

        global_confidence = sum(
            item["weight"]
            * item["confidence"]
            for item in evidence
        )

        return {

            "status":
                "completed",

            "weights":
                normalized_weights,

            "evidence":
                evidence,

            "total_weight":
                float(total_weight),

            "confidence":
                float(
                    np.clip(
                        global_confidence,
                        0.0,
                        1.0
                    )
                )
        }

    # -------------------------------------------------------------------------
    # FUSION BY TASK
    # -------------------------------------------------------------------------

    def group_by_task(
        self,
        results
    ):

        groups = {}

        for result in results:

            task = getattr(
                result,
                "task_type",
                "unknown"
            )

            groups.setdefault(
                task,
                []
            )

            groups[task].append(
                result
            )

        return groups
