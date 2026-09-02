%%writefile coordinator/conflict.py

"""
Conflict Detection and Resolution
=================================

Detects disagreements between agents performing
the same task and resolves them using adaptive trust.
"""

from collections import defaultdict


class ConflictDetector:

    def __init__(
        self,
        confidence_threshold=0.20
    ):
        self.confidence_threshold = float(
            confidence_threshold
        )

    def detect(self, messages):

        conflicts = []

        # Group messages by task
        grouped = defaultdict(list)

        for message in messages:

            if message.status != "success":
                continue

            if message.prediction is None:
                continue

            grouped[message.task_type].append(message)

        # Compare agents solving the same task
        for task_type, task_messages in grouped.items():

            for i in range(len(task_messages)):

                for j in range(i + 1, len(task_messages)):

                    a = task_messages[i]
                    b = task_messages[j]

                    prediction_a = str(a.prediction)
                    prediction_b = str(b.prediction)

                    if prediction_a == prediction_b:
                        continue

                    confidence_gap = abs(
                        float(a.confidence)
                        - float(b.confidence)
                    )

                    severity = (
                        "high"
                        if confidence_gap >= 0.40
                        else "medium"
                        if confidence_gap >= 0.20
                        else "low"
                    )

                    conflicts.append({

                        "task_type": task_type,

                        "agent_a": a.agent_id,

                        "agent_b": b.agent_id,

                        "prediction_a": a.prediction,

                        "prediction_b": b.prediction,

                        "confidence_a": float(
                            a.confidence
                        ),

                        "confidence_b": float(
                            b.confidence
                        ),

                        "trust_a": float(
                            a.trust
                        ),

                        "trust_b": float(
                            b.trust
                        ),

                        "confidence_gap": confidence_gap,

                        "severity": severity
                    })

        return conflicts


class ConflictResolutionEngine:

    def __init__(self, detector=None):

        self.detector = (
            detector
            if detector is not None
            else ConflictDetector()
        )

    def resolve(self, messages):

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

            score_a = (
                float(agent_a.trust)
                * float(agent_a.confidence)
                * float(agent_a.quality)
            )

            score_b = (
                float(agent_b.trust)
                * float(agent_b.confidence)
                * float(agent_b.quality)
            )

            if abs(score_a - score_b) < 0.05:

                resolution = "uncertain"

                selected_agent = None

                selected_prediction = None

            elif score_a > score_b:

                resolution = "agent_a"

                selected_agent = agent_a.agent_id

                selected_prediction = (
                    agent_a.prediction
                )

            else:

                resolution = "agent_b"

                selected_agent = agent_b.agent_id

                selected_prediction = (
                    agent_b.prediction
                )

            resolutions.append({

                "task_type":
                    conflict["task_type"],

                "agent_a":
                    conflict["agent_a"],

                "agent_b":
                    conflict["agent_b"],

                "resolution":
                    resolution,

                "selected_agent":
                    selected_agent,

                "selected_prediction":
                    selected_prediction,

                "score_a":
                    score_a,

                "score_b":
                    score_b,

                "severity":
                    conflict["severity"]
            })

        return {

            "conflicts": conflicts,

            "resolutions": resolutions,

            "num_conflicts":
                len(conflicts)
        }
