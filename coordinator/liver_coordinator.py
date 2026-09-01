# ============================================================
# LIVER AI — MULTI-AGENT COORDINATOR
# Compatible avec les agents actuels du projet
# ============================================================

from coordinator.trust_manager import TrustManager
from coordinator.conflict_detector import ConflictDetector
from coordinator.adaptive_fusion import AdaptiveFusion


class LiverCoordinator:

    def __init__(self, agents):

        self.agents = agents

        self.trust_manager = TrustManager()

        self.conflict_detector = ConflictDetector()

        self.fusion = AdaptiveFusion()

    # ========================================================
    # NORMALIZE AGENT RESULT
    # ========================================================

    def _normalize_result(self, agent, result):

        agent_name = getattr(
            agent,
            "name",
            agent.__class__.__name__
        )

        # ----------------------------------------------------
        # Agent returns a dictionary
        # ----------------------------------------------------

        if isinstance(result, dict):

            normalized = dict(result)

            normalized.setdefault(
                "agent",
                agent_name
            )

            status = normalized.get(
                "status",
                "completed"
            )

            normalized["status"] = status

            # probability -> confidence
            if (
                normalized.get("confidence") is None
                and normalized.get("probability") is not None
            ):

                normalized["confidence"] = float(
                    normalized["probability"]
                )

            # confidence absent
            if normalized.get("confidence") is None:

                normalized["confidence"] = 0.0

            # quality
            if normalized.get("quality") is None:

                normalized["quality"] = 1.0

            # uncertainty
            if normalized.get("uncertainty") is None:

                normalized["uncertainty"] = (
                    1.0 -
                    normalized["confidence"]
                )

            # trust will be calculated later
            normalized.setdefault(
                "trust",
                None
            )

            return normalized

        # ----------------------------------------------------
        # Object-style AgentResult
        # ----------------------------------------------------

        normalized = {

            "agent": getattr(
                result,
                "agent",
                agent_name
            ),

            "prediction": getattr(
                result,
                "prediction",
                None
            ),

            "probability": getattr(
                result,
                "probability",
                getattr(
                    result,
                    "confidence",
                    None
                )
            ),

            "confidence": getattr(
                result,
                "confidence",
                getattr(
                    result,
                    "probability",
                    0.0
                )
            ),

            "uncertainty": getattr(
                result,
                "uncertainty",
                None
            ),

            "quality": getattr(
                result,
                "quality",
                1.0
            ),

            "trust": getattr(
                result,
                "trust",
                None
            ),

            "status": getattr(
                result,
                "status",
                "completed"
            ),

            "error": getattr(
                result,
                "error",
                None
            )
        }

        if normalized["uncertainty"] is None:

            normalized["uncertainty"] = (
                1.0 -
                float(
                    normalized["confidence"] or 0.0
                )
            )

        return normalized

    # ========================================================
    # TRUST
    # ========================================================

    def _compute_trust(self, result):

        confidence = float(
            result.get(
                "confidence",
                0.0
            ) or 0.0
        )

        quality = float(
            result.get(
                "quality",
                1.0
            ) or 1.0
        )

        try:

            trust = self.trust_manager.compute_trust(

                agent_name=result.get(
                    "agent",
                    "UnknownAgent"
                ),

                confidence=confidence,

                quality=quality
            )

            return float(trust)

        except Exception:

            # Fallback simple et sûr
            return confidence * quality

    # ========================================================
    # SIMPLE CONFLICT DETECTION
    # ========================================================

    def _detect_conflicts(self, results):

        valid_results = [

            r for r in results

            if r.get("status") == "completed"
            and r.get("prediction") is not None

        ]

        conflicts = []

        for i in range(
            len(valid_results)
        ):

            for j in range(
                i + 1,
                len(valid_results)
            ):

                a = valid_results[i]

                b = valid_results[j]

                prediction_a = str(
                    a.get("prediction")
                )

                prediction_b = str(
                    b.get("prediction")
                )

                if prediction_a != prediction_b:

                    conflicts.append({

                        "agent_1":
                            a.get("agent"),

                        "prediction_1":
                            prediction_a,

                        "agent_2":
                            b.get("agent"),

                        "prediction_2":
                            prediction_b

                    })

        return conflicts

    # ========================================================
    # SIMPLE FUSION
    # ========================================================

    def _fuse_results(self, results):

        valid_results = [

            r for r in results

            if r.get("status") == "completed"
            and r.get("prediction") is not None

        ]

        if not valid_results:

            return {

                "prediction": None,

                "confidence": 0.0,

                "weighted_votes": {}

            }

        weighted_votes = {}

        for result in valid_results:

            prediction = str(
                result.get(
                    "prediction"
                )
            )

            confidence = float(
                result.get(
                    "confidence",
                    0.0
                ) or 0.0
            )

            trust = float(
                result.get(
                    "trust",
                    confidence
                ) or 0.0
            )

            weight = (
                confidence *
                trust
            )

            weighted_votes[prediction] = (

                weighted_votes.get(
                    prediction,
                    0.0
                )
                +
                weight

            )

        if not weighted_votes:

            return {

                "prediction": None,

                "confidence": 0.0,

                "weighted_votes": {}

            }

        final_prediction = max(
            weighted_votes,
            key=weighted_votes.get
        )

        total_weight = sum(
            weighted_votes.values()
        )

        final_confidence = (

            weighted_votes[
                final_prediction
            ]
            /
            total_weight

            if total_weight > 0
            else 0.0

        )

        return {

            "prediction":
                final_prediction,

            "confidence":
                float(final_confidence),

            "weighted_votes":
                weighted_votes

        }

    # ========================================================
    # RUN
    # ========================================================

    def run(self, patient_data):

        results = []

        # ====================================================
        # 1. EXECUTE AGENTS
        # ====================================================

        for agent in self.agents:

            agent_name = getattr(
                agent,
                "name",
                agent.__class__.__name__
            )

            try:

                result = agent.predict(
                    patient_data
                )

                normalized = (
                    self._normalize_result(
                        agent,
                        result
                    )
                )

                results.append(
                    normalized
                )

            except Exception as e:

                results.append({

                    "agent":
                        agent_name,

                    "prediction":
                        None,

                    "probability":
                        None,

                    "confidence":
                        0.0,

                    "uncertainty":
                        1.0,

                    "quality":
                        0.0,

                    "trust":
                        0.0,

                    "status":
                        "error",

                    "error":
                        str(e)

                })

        # ====================================================
        # 2. COMPUTE TRUST
        # ====================================================

        for result in results:

            if result.get(
                "status"
            ) != "completed":

                continue

            result["trust"] = (
                self._compute_trust(
                    result
                )
            )

        # ====================================================
        # 3. DETECT CONFLICTS
        # ====================================================

        conflicts = (
            self._detect_conflicts(
                results
            )
        )

        # ====================================================
        # 4. FUSION
        # ====================================================

        fusion = (
            self._fuse_results(
                results
            )
        )

        # ====================================================
        # 5. FINAL RESULT
        # ====================================================

        return {

            "status":
                "completed",

            "agents":
                results,

            "conflicts":
                conflicts,

            "fusion":
                fusion

        }
