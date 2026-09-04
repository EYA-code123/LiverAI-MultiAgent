
from typing import Dict, Any
from datetime import datetime


class LiverCoordinator:

    def __init__(self):
        self.agent_weights = {
            "cirrhosis": 1.0,
            "fatty_liver": 1.0,
            "fibrosis": 1.0,
            "tumor": 1.0,
            "tumor_classification": 1.0,
            "segmentation": 1.0,
            "liver_segmentation": 1.0,
            "clinical_reasoning": 1.0,
        }

    # ================================================================
    # MAIN COORDINATION
    # ================================================================

    def coordinate(self, agent_results: Dict[str, Any]) -> Dict[str, Any]:

        normalized = {}

        for agent_name, result in agent_results.items():

            normalized[agent_name] = self._normalize_result(
                agent_name,
                result
            )

        conflicts = self._detect_conflicts(normalized)

        confidence = self._calculate_confidence(normalized)

        return {
            "timestamp": datetime.now().isoformat(),

            "agent_results": normalized,

            "coordination": {
                "agents_received": list(normalized.keys()),

                "successful_agents": [
                    name
                    for name, result in normalized.items()
                    if result["status"] == "success"
                ],

                "failed_agents": [
                    name
                    for name, result in normalized.items()
                    if result["status"] == "error"
                ],

                "unavailable_agents": [
                    name
                    for name, result in normalized.items()
                    if result["status"] == "unavailable"
                ],
            },

            "confidence": confidence,

            "conflicts": conflicts,

            "clinical_context": self._build_clinical_context(
                normalized
            ),
        }

    # ================================================================
    # NORMALIZATION
    # ================================================================

    def _normalize_result(
        self,
        agent_name: str,
        result: Any
    ) -> Dict[str, Any]:

        if result is None:

            return {
                "agent": agent_name,
                "status": "unavailable",
                "prediction": None,
                "probability": None,
                "confidence": 0.0,
            }

        if isinstance(result, dict):

            if result.get("status") == "error":

                return {
                    "agent": agent_name,
                    "status": "error",
                    "prediction": None,
                    "probability": None,
                    "confidence": 0.0,
                    "error": result.get("error"),
                }

            return {
                "agent": agent_name,
                "status": "success",
                "prediction": result.get("prediction"),
                "probability": result.get("probability"),
                "confidence": self._extract_confidence(result),
                "raw_result": result,
            }

        return {
            "agent": agent_name,
            "status": "success",
            "prediction": result,
            "probability": None,
            "confidence": 0.5,
            "raw_result": result,
        }

    # ================================================================
    # CONFIDENCE
    # ================================================================

    def _extract_confidence(self, result):

        confidence = result.get("confidence")

        if confidence is not None:

            try:
                return max(
                    0.0,
                    min(1.0, float(confidence))
                )

            except (ValueError, TypeError):
                pass

        probability = result.get("probability")

        if probability is not None:

            try:

                if isinstance(probability, (list, tuple)):

                    if len(probability) > 0:
                        return float(max(probability))

                return max(
                    0.0,
                    min(1.0, float(probability))
                )

            except (ValueError, TypeError):
                pass

        return 0.5

    # ================================================================
    # GLOBAL CONFIDENCE
    # ================================================================

    def _calculate_confidence(self, results):

        valid_results = [
            result
            for result in results.values()
            if result["status"] == "success"
        ]

        if not valid_results:

            return {
                "global_confidence": 0.0,
                "level": "none",
                "agents_used": 0,
            }

        values = [
            result["confidence"]
            for result in valid_results
        ]

        mean_confidence = sum(values) / len(values)

        if mean_confidence >= 0.80:
            level = "high"

        elif mean_confidence >= 0.60:
            level = "moderate"

        else:
            level = "low"

        return {
            "global_confidence": round(
                mean_confidence,
                4
            ),
            "level": level,
            "agents_used": len(valid_results),
        }

    # ================================================================
    # CONFLICT DETECTION
    # ================================================================

    def _detect_conflicts(self, results):

        conflicts = []

        # ------------------------------------------------------------
        # Cirrhosis vs Fibrosis
        # ------------------------------------------------------------

        cirrhosis = results.get("cirrhosis")
        fibrosis = results.get("fibrosis")

        if (
            cirrhosis
            and fibrosis
            and cirrhosis["status"] == "success"
            and fibrosis["status"] == "success"
        ):

            conflict = self._check_cirrhosis_fibrosis(
                cirrhosis,
                fibrosis
            )

            if conflict is not None:
                conflicts.append(conflict)

        # ------------------------------------------------------------
        # Fatty Liver vs other liver findings
        # ------------------------------------------------------------

        fatty = results.get("fatty_liver")

        if fatty and fatty["status"] == "success":

            conflict = self._check_fatty_liver(
                fatty,
                cirrhosis,
                fibrosis
            )

            if conflict is not None:
                conflicts.append(conflict)

        return conflicts

    # ================================================================
    # CIRRHOSIS / FIBROSIS
    # ================================================================

    def _check_cirrhosis_fibrosis(
        self,
        cirrhosis,
        fibrosis
    ):

        c = cirrhosis.get("prediction")
        f = fibrosis.get("prediction")

        if c is None or f is None:
            return None

        return {
            "type": "cross_agent_comparison",

            "agents": [
                "cirrhosis",
                "fibrosis"
            ],

            "cirrhosis_prediction": c,

            "fibrosis_prediction": f,

            "interpretation":
                "Predictions originate from different models "
                "and should be interpreted jointly rather than "
                "combined by majority voting."
        }

    # ================================================================
    # FATTY LIVER
    # ================================================================

    def _check_fatty_liver(
        self,
        fatty,
        cirrhosis,
        fibrosis
    ):

        return {
            "type": "multimodal_context",

            "agents": [
                "fatty_liver"
            ],

            "interpretation":
                "Fatty liver prediction is treated as an "
                "independent clinical finding."
        }

    # ================================================================
    # CLINICAL CONTEXT
    # ================================================================

    def _build_clinical_context(self, results):

        context = {}

        for agent_name, result in results.items():

            if result["status"] != "success":
                continue

            context[agent_name] = {
                "prediction": result.get("prediction"),
                "probability": result.get("probability"),
                "confidence": result.get("confidence"),
            }

        return context


# ================================================================
# BACKWARD COMPATIBILITY
# ================================================================
#
# coordinator/__init__.py attend :
#
#     from coordinator.coordinator import LiverAICoordinator
#
# Notre classe principale s'appelle LiverCoordinator.
# Cet alias permet donc aux deux noms de fonctionner.
#

LiverAICoordinator = LiverCoordinator
