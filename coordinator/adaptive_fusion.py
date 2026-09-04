# ================================================================
# ADAPTIVE FUSION
# ================================================================
# Combines heterogeneous outputs from the LiverAI agents.
#
# Agents:
#   - Fatty Liver
#   - Fibrosis
#   - Cirrhosis
#   - Tumor Classification
#   - Liver Segmentation
#   - Clinical Reasoning
#
# IMPORTANT:
# The agents do NOT all predict the same target.
# Therefore this module performs evidence aggregation rather than
# blindly averaging unrelated probabilities.
# ================================================================

from typing import Any, Dict, List
import math


class AdaptiveFusion:
    """
    Adaptive evidence fusion for the LiverAI multi-agent system.
    """

    def __init__(
        self,
        min_confidence: float = 0.0,
        min_quality: float = 0.0,
        use_trust: bool = True,
    ):
        self.min_confidence = float(min_confidence)
        self.min_quality = float(min_quality)
        self.use_trust = bool(use_trust)

    # ============================================================
    # PUBLIC API
    # ============================================================

    def fuse(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Fuse outputs from all available agents.

        Parameters
        ----------
        results : list of dict
            Normalized agent results.

        Returns
        -------
        dict
            Fused evidence result.
        """

        if results is None:
            results = []

        if not isinstance(results, list):
            raise TypeError(
                "AdaptiveFusion.fuse() expects a list of dictionaries."
            )

        # --------------------------------------------------------
        # Keep only valid dictionaries
        # --------------------------------------------------------

        valid_results = []

        for result in results:

            if result is None:
                continue

            if hasattr(result, "to_dict"):
                try:
                    result = result.to_dict()
                except Exception:
                    continue

            if not isinstance(result, dict):
                continue

            valid_results.append(result)

        if not valid_results:
            return {
                "status": "unavailable",
                "reason": "No valid agent results",
                "results": [],
                "evidence": {},
                "coverage": 0.0,
                "mean_confidence": 0.0,
                "mean_trust": 0.0,
            }

        # --------------------------------------------------------
        # Process results
        # --------------------------------------------------------

        evidence = {}

        successful = []
        failed = []
        not_run = []

        for result in valid_results:

            agent_id = str(
                result.get("agent_id", "unknown")
            )

            status = str(
                result.get("status", "unknown")
            ).lower()

            if status == "success":

                successful.append(result)

            elif status == "not_run":

                not_run.append(result)

            else:

                failed.append(result)

            # -----------------------------------------------
            # Extract evidence
            # -----------------------------------------------

            evidence[agent_id] = self._extract_evidence(result)

        # --------------------------------------------------------
        # Metrics
        # --------------------------------------------------------

        total_agents = len(valid_results)

        successful_count = len(successful)

        coverage = (
            successful_count / total_agents
            if total_agents > 0
            else 0.0
        )

        confidences = []

        trusts = []

        qualities = []

        for result in successful:

            confidence = self._safe_float(
                result.get("confidence")
            )

            trust = self._safe_float(
                result.get("trust")
            )

            quality = self._safe_float(
                result.get("quality")
            )

            if confidence is not None:
                confidences.append(
                    self._clip01(confidence)
                )

            if trust is not None:
                trusts.append(
                    self._clip01(trust)
                )

            if quality is not None:
                qualities.append(
                    self._clip01(quality)
                )

        mean_confidence = (
            sum(confidences) / len(confidences)
            if confidences
            else 0.0
        )

        mean_trust = (
            sum(trusts) / len(trusts)
            if trusts
            else 0.0
        )

        mean_quality = (
            sum(qualities) / len(qualities)
            if qualities
            else 0.0
        )

        # --------------------------------------------------------
        # Weighted evidence score
        # --------------------------------------------------------

        weighted_evidence = self._calculate_weighted_evidence(
            successful
        )

        # --------------------------------------------------------
        # Status
        # --------------------------------------------------------

        if successful_count == 0:

            status = "unavailable"

        elif coverage < 1.0:

            status = "partial"

        else:

            status = "success"

        # --------------------------------------------------------
        # Final result
        # --------------------------------------------------------

        return {
            "status": status,

            "results": valid_results,

            "evidence": evidence,

            "coverage": coverage,

            "successful_agents": successful_count,

            "total_agents": total_agents,

            "failed_agents": len(failed),

            "not_run_agents": len(not_run),

            "mean_confidence": mean_confidence,

            "mean_trust": mean_trust,

            "mean_quality": mean_quality,

            "weighted_evidence": weighted_evidence,

            "fusion_method": (
                "trust_weighted_heterogeneous_evidence"
            ),
        }

    # ============================================================
    # EVIDENCE EXTRACTION
    # ============================================================

    def _extract_evidence(
        self,
        result: Dict[str, Any]
    ) -> Dict[str, Any]:

        agent_id = result.get(
            "agent_id",
            "unknown"
        )

        task_type = result.get(
            "task_type",
            agent_id
        )

        prediction = result.get(
            "prediction"
        )

        probability = result.get(
            "probability"
        )

        confidence = result.get(
            "confidence",
            0.0
        )

        uncertainty = result.get(
            "uncertainty",
            1.0
        )

        quality = result.get(
            "quality",
            0.0
        )

        trust = result.get(
            "trust",
            0.0
        )

        class_probabilities = result.get(
            "class_probabilities"
        )

        modality = result.get(
            "modality"
        )

        status = result.get(
            "status",
            "unknown"
        )

        evidence = {
            "agent_id": agent_id,
            "task_type": task_type,
            "status": status,
            "prediction": prediction,
            "probability": self._safe_float(
                probability
            ),
            "confidence": self._clip01(
                self._safe_float(
                    confidence,
                    default=0.0
                )
            ),
            "uncertainty": self._clip01(
                self._safe_float(
                    uncertainty,
                    default=1.0
                )
            ),
            "quality": self._clip01(
                self._safe_float(
                    quality,
                    default=0.0
                )
            ),
            "trust": self._clip01(
                self._safe_float(
                    trust,
                    default=0.0
                )
            ),
            "class_probabilities": class_probabilities,
            "modality": modality,
        }

        # --------------------------------------------------------
        # Preserve useful details
        # --------------------------------------------------------

        if "details" in result:
            evidence["details"] = result["details"]

        if "explanation" in result:
            evidence["explanation"] = result["explanation"]

        if "error" in result:
            evidence["error"] = result["error"]

        return evidence

    # ============================================================
    # WEIGHTED EVIDENCE
    # ============================================================

    def _calculate_weighted_evidence(
        self,
        results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Calculate reliability weight for each successful agent.

        Weight is based on:
            trust
            confidence
            quality
            uncertainty
            missing data
        """

        weighted = []

        for result in results:

            if str(
                result.get("status", "")
            ).lower() != "success":

                continue

            confidence = self._clip01(
                self._safe_float(
                    result.get(
                        "confidence"
                    ),
                    default=0.0
                )
            )

            uncertainty = self._clip01(
                self._safe_float(
                    result.get(
                        "uncertainty"
                    ),
                    default=1.0
                )
            )

            quality = self._clip01(
                self._safe_float(
                    result.get(
                        "quality"
                    ),
                    default=0.0
                )
            )

            trust = self._clip01(
                self._safe_float(
                    result.get(
                        "trust"
                    ),
                    default=0.0
                )
            )

            missing_ratio = self._clip01(
                self._safe_float(
                    result.get(
                        "missing_data_ratio"
                    ),
                    default=0.0
                )
            )

            # ----------------------------------------------------
            # Reliability
            # ----------------------------------------------------

            reliability = (
                0.40 * trust
                + 0.30 * confidence
                + 0.20 * quality
                + 0.10 * (1.0 - uncertainty)
            )

            # Penalize missing information
            reliability *= (
                1.0 - 0.5 * missing_ratio
            )

            reliability = self._clip01(
                reliability
            )

            weighted.append(
                {
                    "agent_id": result.get(
                        "agent_id"
                    ),
                    "task_type": result.get(
                        "task_type"
                    ),
                    "prediction": result.get(
                        "prediction"
                    ),
                    "confidence": confidence,
                    "uncertainty": uncertainty,
                    "quality": quality,
                    "trust": trust,
                    "missing_data_ratio": missing_ratio,
                    "weight": reliability,
                }
            )

        return weighted

    # ============================================================
    # UTILITY FUNCTIONS
    # ============================================================

    @staticmethod
    def _safe_float(
        value: Any,
        default=None
    ):

        if value is None:
            return default

        try:

            value = float(value)

            if not math.isfinite(value):
                return default

            return value

        except (
            TypeError,
            ValueError
        ):

            return default

    @staticmethod
    def _clip01(
        value: Any
    ) -> float:

        try:

            value = float(value)

        except (
            TypeError,
            ValueError
        ):

            return 0.0

        if not math.isfinite(value):
            return 0.0

        return max(
            0.0,
            min(
                1.0,
                value
            )
        )
