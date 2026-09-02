"""
Conflict Resolution Intelligence
=================================

Detects disagreements between agents solving the SAME task.

Cross-task differences are not considered conflicts because
cirrhosis stage, fibrosis, tumor type and segmentation are
different clinical outputs.
"""

import numpy as np


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

        successful = [
            m for m in messages
            if m.status == "success"
            and m.prediction is not None
        ]

        for i in range(len(successful)):

            for j in range(i + 1, len(successful)):

                a = successful[i]
                b = successful[j]

                # Different tasks = no direct conflict
                if a.task_type != b.task_type:
                    continue

                prediction_a = str(
                    a.prediction
                )

                prediction_b = str(
                    b.prediction
                )

                if prediction_a == prediction_b:
                    continue

                confidence_gap = abs(
                    a.confidence
                    -
                    b.confidence
                )

                conflicts.append({
                    "agent_a": a.agent_id,
                    "agent_b": b.agent_id,
                    "task_type": a.task_type,
                    "prediction_a": prediction_a,
                    "prediction_b": prediction_b,
                    "confidence_a": a.confidence,
                    "confidence_b": b.confidence,
                    "confidence_gap": confidence_gap,
                    "severity": self._severity(
                        confidence_gap
                    )
                })

        return conflicts

    @staticmethod
    def _severity(
        confidence_gap
    ):

        if confidence_gap < 0.10:
            return "high"

        if confidence_gap < 0.25:
            return "medium"

        return "low"


class ConflictResolutionEngine:

    def __init__(
        self,
        detector=None
    ):

        self.detector = (
            detector
            or ConflictDetector()
        )

    def resolve(
        self,
        messages
    ):

        conflicts = self.detector.detect(
            messages
        )

        resolutions = []

        for conflict in conflicts:

            agent_a = next(
                (
                    m for m in messages
                    if m.agent_id
                    == conflict["agent_a"]
                ),
                None
            )

            agent_b = next(
                (
                    m for m in messages
                    if m.agent_id
                    == conflict["agent_b"]
                ),
                None
            )

            if agent_a is None or agent_b is None:
                continue

            # Adaptive trust-confidence score
            score_a = (
                agent_a.trust
                * agent_a.confidence
                * agent_a.quality
            )

            score_b = (
                agent_b.trust
                * agent_b.confidence
                * agent_b.quality
            )

            if abs(score_a - score_b) < 0.05:

                resolution = "uncertain"

            elif score_a > score_b:

                resolution = agent_a.agent_id

            else:

                resolution = agent_b.agent_id

            resolutions.append({
                **conflict,
                "score_a": float(score_a),
                "score_b": float(score_b),
                "resolution": resolution
            })

        return {
            "num_conflicts": len(conflicts),
            "conflicts": conflicts,
            "resolutions": resolutions
        }
