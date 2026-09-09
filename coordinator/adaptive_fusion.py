# ============================================================
# coordinator/adaptive_fusion.py
# Adaptive Evidence Fusion
# ============================================================

from typing import Any, Dict, List
import math


class AdaptiveFusion:
    """
    Adaptive evidence fusion for the LiverAI multi-agent system.

    IMPORTANT
    ---------
    LiverAI contains heterogeneous agents.

    Examples:
        cirrhosis
        fatty_liver_classification
        fibrosis_classification
        tumor_classification
        liver_segmentation
        clinical_reasoning

    These tasks must NOT be blindly fused together.

    Therefore this class performs:

        1. Evidence extraction
        2. Reliability weighting
        3. Same-task fusion
        4. Global heterogeneous evidence aggregation
    """

    # ========================================================
    # INITIALIZATION
    # ========================================================

    def __init__(
        self,
        min_confidence: float = 0.0,
        min_quality: float = 0.0,
        use_trust: bool = True,
    ):

        self.min_confidence = float(
            min_confidence
        )

        self.min_quality = float(
            min_quality
        )

        self.use_trust = bool(
            use_trust
        )

    # ========================================================
    # PUBLIC API
    # ========================================================

    def fuse(
        self,
        results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Fuse heterogeneous agent outputs.

        Same-task outputs are fused together.

        Different tasks remain separate.

        Parameters
        ----------
        results:
            List of normalized agent results.

        Returns
        -------
        dict
        """

        if results is None:
            results = []

        if not isinstance(
            results,
            list
        ):

            raise TypeError(
                "AdaptiveFusion.fuse() expects "
                "a list of dictionaries."
            )

        # ----------------------------------------------------
        # Normalize result objects
        # ----------------------------------------------------

        valid_results = []

        for result in results:

            if result is None:
                continue

            # Support AgentMessage-like objects
            if hasattr(
                result,
                "to_dict"
            ):

                try:
                    result = result.to_dict()

                except Exception:
                    continue

            if not isinstance(
                result,
                dict
            ):
                continue

            valid_results.append(
                result
            )

        # ----------------------------------------------------
        # No results
        # ----------------------------------------------------

        if not valid_results:

            return {

                "status":
                    "unavailable",

                "reason":
                    "No valid agent results",

                "results":
                    [],

                "evidence":
                    {},

                "same_task_fusion":
                    {},

                "task_groups":
                    {},

                "coverage":
                    0.0,

                "successful_agents":
                    0,

                "total_agents":
                    0,

                "failed_agents":
                    0,

                "not_run_agents":
                    0,

                "mean_confidence":
                    0.0,

                "mean_trust":
                    0.0,

                "mean_quality":
                    0.0,

                "weighted_evidence":
                    [],

                "fusion_method":
                    "trust_weighted_same_task_evidence",
            }

        # ====================================================
        # CLASSIFY RESULTS
        # ====================================================

        evidence = {}

        successful = []

        failed = []

        not_run = []

        for result in valid_results:

            agent_id = str(
                result.get(
                    "agent_id",
                    "unknown"
                )
            )

            status = str(
                result.get(
                    "status",
                    "unknown"
                )
            ).lower()

            if status == "success":

                successful.append(
                    result
                )

            elif status == "not_run":

                not_run.append(
                    result
                )

            else:

                failed.append(
                    result
                )

            evidence[
                agent_id
            ] = self._extract_evidence(
                result
            )

        # ====================================================
        # GROUP BY TASK
        # ====================================================

        task_groups = (
            self._group_by_task(
                successful
            )
        )

        # ====================================================
        # SAME-TASK FUSION
        # ====================================================

        same_task_fusion = {}

        for task_type, task_results in (
            task_groups.items()
        ):

            same_task_fusion[
                task_type
            ] = self._fuse_same_task(
                task_type,
                task_results
            )

        # ====================================================
        # METRICS
        # ====================================================

        total_agents = len(
            valid_results
        )

        successful_count = len(
            successful
        )

        coverage = (
            successful_count /
            total_agents
            if total_agents > 0
            else 0.0
        )

        confidences = []

        trusts = []

        qualities = []

        for result in successful:

            confidence = self._safe_float(
                result.get(
                    "confidence"
                )
            )

            trust = self._safe_float(
                result.get(
                    "trust"
                )
            )

            quality = self._safe_float(
                result.get(
                    "quality"
                )
            )

            if confidence is not None:

                confidences.append(
                    self._clip01(
                        confidence
                    )
                )

            if trust is not None:

                trusts.append(
                    self._clip01(
                        trust
                    )
                )

            if quality is not None:

                qualities.append(
                    self._clip01(
                        quality
                    )
                )

        mean_confidence = (
            sum(confidences) /
            len(confidences)
            if confidences
            else 0.0
        )

        mean_trust = (
            sum(trusts) /
            len(trusts)
            if trusts
            else 0.0
        )

        mean_quality = (
            sum(qualities) /
            len(qualities)
            if qualities
            else 0.0
        )

        # ====================================================
        # WEIGHTED EVIDENCE
        # ====================================================

        weighted_evidence = (
            self._calculate_weighted_evidence(
                successful
            )
        )

        # ====================================================
        # STATUS
        # ====================================================

        if successful_count == 0:

            status = "unavailable"

        elif coverage < 1.0:

            status = "partial"

        else:

            status = "success"

        # ====================================================
        # FINAL OUTPUT
        # ====================================================

        return {

            "status":
                status,

            "results":
                valid_results,

            "evidence":
                evidence,

            # ------------------------------------------------
            # IMPORTANT FOR TESTS AND COORDINATOR
            # ------------------------------------------------

            "same_task_fusion":
                same_task_fusion,

            "task_groups":
                {
                    task_type:
                        [
                            r.get(
                                "agent_id"
                            )
                            for r in task_results
                        ]
                    for task_type, task_results
                    in task_groups.items()
                },

            "coverage":
                coverage,

            "successful_agents":
                successful_count,

            "total_agents":
                total_agents,

            "failed_agents":
                len(failed),

            "not_run_agents":
                len(not_run),

            "mean_confidence":
                mean_confidence,

            "mean_trust":
                mean_trust,

            "mean_quality":
                mean_quality,

            "weighted_evidence":
                weighted_evidence,

            "fusion_method":
                (
                    "trust_weighted_"
                    "same_task_evidence"
                ),
        }

    # ========================================================
    # GROUP BY TASK
    # ========================================================

    def _group_by_task(
        self,
        results: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:

        groups = {}

        for result in results:

            task_type = str(
                result.get(
                    "task_type",
                    result.get(
                        "task",
                        "unknown"
                    )
                )
            )

            groups.setdefault(
                task_type,
                []
            ).append(
                result
            )

        return groups

    # ========================================================
    # SAME TASK FUSION
    # ========================================================

    def _fuse_same_task(
        self,
        task_type: str,
        results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:

        if not results:

            return {

                "task_type":
                    task_type,

                "status":
                    "unavailable",

                "num_agents":
                    0,

                "prediction":
                    None,

                "confidence":
                    0.0,

                "trust":
                    0.0,

                "quality":
                    0.0,

                "agreement":
                    0.0,

                "weights":
                    {},
            }

        # ----------------------------------------------------
        # Calculate reliability weights
        # ----------------------------------------------------

        weights = {}

        for result in results:

            agent_id = str(
                result.get(
                    "agent_id",
                    "unknown"
                )
            )

            weights[
                agent_id
            ] = self._calculate_reliability(
                result
            )

        total_weight = sum(
            weights.values()
        )

        # Prevent division by zero
        if total_weight <= 0.0:

            total_weight = float(
                len(results)
            )

            weights = {
                str(
                    result.get(
                        "agent_id",
                        "unknown"
                    )
                ):
                    1.0
                for result in results
            }

        # ----------------------------------------------------
        # Prediction voting
        # ----------------------------------------------------

        prediction_scores = {}

        for result in results:

            prediction = result.get(
                "prediction"
            )

            if prediction is None:
                continue

            key = str(
                prediction
            )

            weight = weights.get(
                str(
                    result.get(
                        "agent_id",
                        "unknown"
                    )
                ),
                0.0
            )

            prediction_scores[
                key
            ] = (
                prediction_scores.get(
                    key,
                    0.0
                )
                +
                weight
            )

        # ----------------------------------------------------
        # Best prediction
        # ----------------------------------------------------

        if prediction_scores:

            best_prediction = max(
                prediction_scores,
                key=prediction_scores.get
            )

            prediction_confidence = (
                prediction_scores[
                    best_prediction
                ]
                /
                total_weight
            )

        else:

            best_prediction = None

            prediction_confidence = 0.0

        # ----------------------------------------------------
        # Mean metrics
        # ----------------------------------------------------

        confidence_values = []

        trust_values = []

        quality_values = []

        for result in results:

            confidence_values.append(
                self._clip01(
                    result.get(
                        "confidence",
                        0.0
                    )
                )
            )

            trust_values.append(
                self._clip01(
                    result.get(
                        "trust",
                        0.0
                    )
                )
            )

            quality_values.append(
                self._clip01(
                    result.get(
                        "quality",
                        0.0
                    )
                )
            )

        mean_confidence = (
            sum(
                confidence_values
            )
            /
            len(
                confidence_values
            )
            if confidence_values
            else 0.0
        )

        mean_trust = (
            sum(
                trust_values
            )
            /
            len(
                trust_values
            )
            if trust_values
            else 0.0
        )

        mean_quality = (
            sum(
                quality_values
            )
            /
            len(
                quality_values
            )
            if quality_values
            else 0.0
        )

        # ----------------------------------------------------
        # Agreement
        # ----------------------------------------------------

        predictions = [

            str(
                result.get(
                    "prediction"
                )
            )

            for result in results

            if result.get(
                "prediction"
            ) is not None
        ]

        if predictions:

            majority_count = max(
                (
                    predictions.count(
                        prediction
                    )
                    for prediction
                    in set(predictions)
                ),
                default=0
            )

            agreement = (
                majority_count /
                len(predictions)
            )

        else:

            # Segmentation and other non-class outputs
            agreement = 1.0

        # ----------------------------------------------------
        # Segmentation
        # ----------------------------------------------------

        if task_type == "liver_segmentation":

            best_result = max(
                results,
                key=lambda r:
                    self._calculate_reliability(
                        r
                    )
            )

            best_prediction = None

            prediction_confidence = (
                self._clip01(
                    best_result.get(
                        "confidence",
                        0.0
                    )
                )
            )

            agreement = 1.0

        # ----------------------------------------------------
        # Final same-task result
        # ----------------------------------------------------

        return {

            "task_type":
                task_type,

            "status":
                "success",

            "num_agents":
                len(results),

            "prediction":
                best_prediction,

            "confidence":
                self._clip01(
                    prediction_confidence
                ),

            "mean_confidence":
                self._clip01(
                    mean_confidence
                ),

            "trust":
                self._clip01(
                    mean_trust
                ),

            "quality":
                self._clip01(
                    mean_quality
                ),

            "agreement":
                self._clip01(
                    agreement
                ),

            "weights":
                weights,

            "prediction_scores":
                prediction_scores,

            "agents":
                [
                    result.get(
                        "agent_id"
                    )
                    for result in results
                ],
        }

    # ========================================================
    # EXTRACT EVIDENCE
    # ========================================================

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
            result.get(
                "task",
                agent_id
            )
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

        class_probabilities = (
            result.get(
                "class_probabilities"
            )
        )

        modality = result.get(
            "modality"
        )

        status = result.get(
            "status",
            "unknown"
        )

        evidence = {

            "agent_id":
                agent_id,

            "task_type":
                task_type,

            "status":
                status,

            "prediction":
                prediction,

            "probability":
                self._safe_float(
                    probability
                ),

            "confidence":
                self._clip01(
                    self._safe_float(
                        confidence,
                        default=0.0
                    )
                ),

            "uncertainty":
                self._clip01(
                    self._safe_float(
                        uncertainty,
                        default=1.0
                    )
                ),

            "quality":
                self._clip01(
                    self._safe_float(
                        quality,
                        default=0.0
                    )
                ),

            "trust":
                self._clip01(
                    self._safe_float(
                        trust,
                        default=0.0
                    )
                ),

            "class_probabilities":
                class_probabilities,

            "modality":
                modality,
        }

        # ----------------------------------------------------
        # Preserve details
        # ----------------------------------------------------

        if "details" in result:

            evidence[
                "details"
            ] = result[
                "details"
            ]

        if "explanation" in result:

            evidence[
                "explanation"
            ] = result[
                "explanation"
            ]

        if "error" in result:

            evidence[
                "error"
            ] = result[
                "error"
            ]

        return evidence

    # ========================================================
    # RELIABILITY
    # ========================================================

    def _calculate_reliability(
        self,
        result: Dict[str, Any]
    ) -> float:

        confidence = self._clip01(
            result.get(
                "confidence",
                0.0
            )
        )

        uncertainty = self._clip01(
            result.get(
                "uncertainty",
                1.0
            )
        )

        quality = self._clip01(
            result.get(
                "quality",
                0.0
            )
        )

        trust = self._clip01(
            result.get(
                "trust",
                0.0
            )
        )

        missing_ratio = self._clip01(
            result.get(
                "missing_data_ratio",
                0.0
            )
        )

        # ----------------------------------------------------
        # Adaptive reliability
        # ----------------------------------------------------

        if self.use_trust:

            reliability = (

                0.40 * trust

                +

                0.30 * confidence

                +

                0.20 * quality

                +

                0.10 *
                (
                    1.0 -
                    uncertainty
                )
            )

        else:

            reliability = (

                0.50 * confidence

                +

                0.30 * quality

                +

                0.20 *
                (
                    1.0 -
                    uncertainty
                )
            )

        # ----------------------------------------------------
        # Missing-data penalty
        # ----------------------------------------------------

        reliability *= (
            1.0 -
            0.5 *
            missing_ratio
        )

        # ----------------------------------------------------
        # Minimum thresholds
        # ----------------------------------------------------

        if confidence < self.min_confidence:

            reliability *= 0.5

        if quality < self.min_quality:

            reliability *= 0.5

        return self._clip01(
            reliability
        )

    # ========================================================
    # WEIGHTED EVIDENCE
    # ========================================================

    def _calculate_weighted_evidence(
        self,
        results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:

        weighted = []

        for result in results:

            if str(
                result.get(
                    "status",
                    ""
                )
            ).lower() != "success":

                continue

            confidence = self._clip01(
                result.get(
                    "confidence",
                    0.0
                )
            )

            uncertainty = self._clip01(
                result.get(
                    "uncertainty",
                    1.0
                )
            )

            quality = self._clip01(
                result.get(
                    "quality",
                    0.0
                )
            )

            trust = self._clip01(
                result.get(
                    "trust",
                    0.0
                )
            )

            missing_ratio = self._clip01(
                result.get(
                    "missing_data_ratio",
                    0.0
                )
            )

            reliability = (
                self._calculate_reliability(
                    result
                )
            )

            weighted.append({

                "agent_id":
                    result.get(
                        "agent_id"
                    ),

                "task_type":
                    result.get(
                        "task_type"
                    ),

                "prediction":
                    result.get(
                        "prediction"
                    ),

                "confidence":
                    confidence,

                "uncertainty":
                    uncertainty,

                "quality":
                    quality,

                "trust":
                    trust,

                "missing_data_ratio":
                    missing_ratio,

                "weight":
                    reliability,
            })

        return weighted

    # ========================================================
    # UTILITY
    # ========================================================

    @staticmethod
    def _safe_float(
        value: Any,
        default=None
    ):

        if value is None:
            return default

        try:

            value = float(
                value
            )

            if not math.isfinite(
                value
            ):

                return default

            return value

        except (
            TypeError,
            ValueError
        ):

            return default

    # ========================================================

    @staticmethod
    def _clip01(
        value: Any
    ) -> float:

        try:

            value = float(
                value
            )

        except (
            TypeError,
            ValueError
        ):

            return 0.0

        if not math.isfinite(
            value
        ):

            return 0.0

        return max(
            0.0,
            min(
                1.0,
                value
            )
        )


# ============================================================
# EXPORT
# ============================================================

__all__ = [
    "AdaptiveFusion"
]
