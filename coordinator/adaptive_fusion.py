"""
Adaptive Fusion Intelligence
============================

Task-aware dynamic fusion.

IMPORTANT:
Heterogeneous medical tasks are NOT directly voted together.

Examples:

Cirrhosis classification
        -> classification fusion

Fatty liver classification
        -> classification fusion

Fibrosis
        -> regression fusion

Tumor classification
        -> image classification fusion

Segmentation
        -> segmentation quality fusion

Cross-task synthesis is performed later by the reasoning layer.
"""

import numpy as np
from collections import defaultdict


class AdaptiveFusion:

    def __init__(
        self,
        temperature=1.0
    ):

        self.temperature = temperature

    @staticmethod
    def clip(value):

        return float(
            np.clip(
                float(value),
                0.0,
                1.0
            )
        )

    def compute_dynamic_weight(
        self,
        message
    ):

        trust = self.clip(
            message.trust
        )

        confidence = self.clip(
            message.confidence
        )

        quality = self.clip(
            message.quality
        )

        uncertainty = self.clip(
            message.uncertainty
        )

        agreement = self.clip(
            message.agreement
        )

        stability = self.clip(
            message.stability
        )

        utility = self.clip(
            message.utility
        )

        weight = (
            0.30 * trust
            +
            0.20 * confidence
            +
            0.15 * quality
            +
            0.10 * (1.0 - uncertainty)
            +
            0.10 * agreement
            +
            0.10 * stability
            +
            0.05 * utility
        )

        return max(
            0.000001,
            weight
        )

    def _classification_fusion(
        self,
        messages
    ):

        votes = defaultdict(float)

        weights = {}

        for message in messages:

            weight = self.compute_dynamic_weight(
                message
            )

            weights[
                message.agent_id
            ] = weight

            probabilities = (
                message.class_probabilities
            )

            if probabilities:

                for label, probability in probabilities.items():

                    votes[str(label)] += (
                        weight
                        * float(probability)
                    )

            elif message.prediction is not None:

                votes[
                    str(message.prediction)
                ] += weight

        if not votes:
            return None

        total = sum(votes.values())

        normalized = {
            key: value / total
            for key, value in votes.items()
        }

        prediction = max(
            normalized,
            key=normalized.get
        )

        confidence = normalized[
            prediction
        ]

        return {
            "task_type": "classification",
            "prediction": prediction,
            "confidence": confidence,
            "probabilities": normalized,
            "weights": weights,
        }

    def _regression_fusion(
        self,
        messages
    ):

        values = []
        weights = {}

        for message in messages:

            if message.prediction is None:
                continue

            try:
                value = float(
                    message.prediction
                )
            except Exception:
                continue

            weight = self.compute_dynamic_weight(
                message
            )

            values.append(
                (value, weight)
            )

            weights[
                message.agent_id
            ] = weight

        if not values:
            return None

        total_weight = sum(
            weight
            for _, weight in values
        )

        prediction = sum(
            value * weight
            for value, weight in values
        ) / max(
            total_weight,
            1e-12
        )

        # Weighted disagreement
        variance = sum(
            weight * (value - prediction) ** 2
            for value, weight in values
        ) / max(
            total_weight,
            1e-12
        )

        uncertainty = float(
            np.sqrt(max(variance, 0.0))
        )

        confidence = 1.0 / (
            1.0 + uncertainty
        )

        return {
            "task_type": "regression",
            "prediction": float(prediction),
            "confidence": self.clip(confidence),
            "uncertainty": uncertainty,
            "weights": weights,
        }

    def _segmentation_fusion(
        self,
        messages
    ):

        successful = [
            m for m in messages
            if m.status == "success"
        ]

        if not successful:
            return None

        weights = {
            m.agent_id:
            self.compute_dynamic_weight(m)
            for m in successful
        }

        total = sum(weights.values())

        average_quality = sum(
            m.quality * weights[m.agent_id]
            for m in successful
        ) / max(total, 1e-12)

        return {
            "task_type": "segmentation",
            "prediction": successful[0].prediction,
            "confidence": self.clip(
                average_quality
            ),
            "segmentation_quality": self.clip(
                average_quality
            ),
            "weights": weights,
        }

    def fuse_task(
        self,
        messages,
        task_type
    ):

        if not messages:
            return None

        if task_type == "classification":
            return self._classification_fusion(
                messages
            )

        if task_type in (
            "regression",
            "prediction"
        ):
            return self._regression_fusion(
                messages
            )

        if task_type == "segmentation":
            return self._segmentation_fusion(
                messages
            )

        # Reasoning is not numerically fused
        if task_type == "reasoning":

            return {
                "task_type": "reasoning",
                "prediction": None,
                "confidence": max(
                    (
                        m.confidence
                        for m in messages
                    ),
                    default=0.0
                ),
                "weights": {
                    m.agent_id:
                    self.compute_dynamic_weight(m)
                    for m in messages
                }
            }

        return None

    def fuse(
        self,
        messages
    ):

        """
        Main entry point.

        Groups messages by task type before fusion.
        """

        groups = defaultdict(list)

        for message in messages:

            if message.status != "success":
                continue

            if message.prediction is None:
                continue

            groups[
                message.task_type
            ].append(message)

        fused = {}

        for task_type, group in groups.items():

            result = self.fuse_task(
                group,
                task_type
            )

            if result is not None:

                fused[task_type] = result

        return fused
