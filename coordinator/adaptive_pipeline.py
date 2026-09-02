"""
Adaptive Coordination Intelligence Pipeline
============================================

Implements the complete coordination pipeline:

Phase 1  - Agent communication
Phase 2  - Agent assessment
Phase 4  - Adaptive trust
Phase 5  - Adaptive fusion
Phase 6  - Reasoning
Phase 7  - Conflict resolution
Phase 8  - Decision intelligence
Phase 9  - Action intelligence
Phase 10 - Feedback intelligence

Phase 3 from the original guide is intentionally skipped here
because the supplied guide jumps from Phase 2 to Phase 4.
"""

import time

from coordinator.agent_adapter import AgentAdapter
from coordinator.trust import TrustManager
from coordinator.adaptive_fusion import AdaptiveFusion
from coordinator.conflict import (
    ConflictDetector,
    ConflictResolutionEngine
)
from coordinator.reasoning import EvidenceReasoner
from coordinator.decision import DecisionEngine
from coordinator.action import ActionIntelligence
from coordinator.feedback import FeedbackIntelligence


class AdaptiveCoordinationPipeline:

    def __init__(self):

        # Phase 1
        self.adapter = AgentAdapter()

        # Phase 2 + Phase 4
        self.trust_manager = TrustManager()

        # Phase 5
        self.fusion = AdaptiveFusion()

        # Phase 7
        self.conflict_detector = (
            ConflictDetector()
        )

        self.conflict_engine = (
            ConflictResolutionEngine(
                self.conflict_detector
            )
        )

        # Phase 6
        self.reasoner = EvidenceReasoner()

        # Phase 8
        self.decision_engine = DecisionEngine()

        # Phase 9
        self.action_engine = (
            ActionIntelligence()
        )

        # Phase 10
        self.feedback_engine = (
            FeedbackIntelligence(
                self.trust_manager
            )
        )

    def assess_agents(
        self,
        messages
    ):

        """
        Phase 2 + Phase 4.

        Computes patient-specific trust.
        """

        # First determine agreement inside
        # each task group.

        task_predictions = {}

        for message in messages:

            task_predictions.setdefault(
                message.task_type,
                []
            ).append(
                message.prediction
            )

        for message in messages:

            predictions = task_predictions.get(
                message.task_type,
                []
            )

            if len(predictions) <= 1:

                agreement = 0.5

            else:

                same = sum(
                    1
                    for prediction in predictions
                    if str(prediction)
                    ==
                    str(message.prediction)
                )

                agreement = (
                    same / len(predictions)
                )

            # Stability is estimated from
            # confidence/uncertainty unless
            # historical repeated predictions
            # are available.

            stability = (
                1.0
                - message.uncertainty
            )

            utility = (
                0.5 * message.confidence
                +
                0.5 * message.quality
            )

            self.trust_manager.compute_message_trust(
                message,
                agreement=agreement,
                stability=stability,
                utility=utility,
                modality_available=(
                    message.modality
                    != "unknown"
                )
            )

        return messages

    def run(
        self,
        raw_results,
        patient_id=None,
        agents=None,
        input_data=None,
        ground_truths=None
    ):

        start = time.perf_counter()

        agents = agents or {}

        # =====================================================
        # PHASE 1
        # Unified Agent Messages
        # =====================================================

        messages = self.adapter.adapt_many(
            results=raw_results,
            patient_id=patient_id,
            agents=agents,
            input_data=input_data
        )

        # =====================================================
        # PHASE 2 + 4
        # Agent Assessment + Adaptive Trust
        # =====================================================

        messages = self.assess_agents(
            messages
        )

        # =====================================================
        # PHASE 7
        # Conflict Detection
        # =====================================================

        conflict_result = (
            self.conflict_engine.resolve(
                messages
            )
        )

        conflicts = (
            conflict_result["conflicts"]
        )

        # =====================================================
        # PHASE 5
        # Adaptive Task-Aware Fusion
        # =====================================================

        fused_results = self.fusion.fuse(
            messages
        )

        # =====================================================
        # PHASE 6
        # Evidence Reasoning
        # =====================================================

        reasoning = self.reasoner.synthesize(
            fused_results,
            conflicts
        )

        # =====================================================
        # PHASE 8
        # Confidence-Aware Decision
        # =====================================================

        decisions = self.decision_engine.decide(
            agent_results=messages,
            conflicts=conflicts,
            fused_results=fused_results
        )

        # =====================================================
        # PHASE 9
        # Action Intelligence
        # =====================================================

        actions = self.action_engine.generate(
            decisions,
            reasoning
        )

        # =====================================================
        # PHASE 10
        # Feedback Intelligence
        # =====================================================

        feedback = None

        if ground_truths is not None:

            feedback = (
                self.feedback_engine.update(
                    messages,
                    ground_truths
                )
            )

        total_latency = (
            time.perf_counter()
            - start
        ) * 1000.0

        # =====================================================
        # FINAL COORDINATION OUTPUT
        # =====================================================

        return {

            "patient_id":
                patient_id,

            "coordination": {

                "status":
                    "completed",

                "latency_ms":
                    total_latency,

                "num_agents":
                    len(messages),

                "successful_agents":
                    sum(
                        1
                        for message in messages
                        if message.status
                        == "success"
                    ),

                "failed_agents":
                    sum(
                        1
                        for message in messages
                        if message.status
                        != "success"
                    )
            },

            # Phase 1
            "agent_messages": [
                message.to_dict()
                for message in messages
            ],

            # Phase 2 / 4
            "agent_trust": {
                message.agent_id:
                    message.trust
                for message in messages
            },

            # Phase 5
            "adaptive_fusion":
                fused_results,

            # Phase 7
            "conflict_resolution":
                conflict_result,

            # Phase 6
            "reasoning":
                reasoning,

            # Phase 8
            "decision":
                decisions,

            # Phase 9
            "actions":
                actions,

            # Phase 10
            "feedback":
                feedback
        } 
