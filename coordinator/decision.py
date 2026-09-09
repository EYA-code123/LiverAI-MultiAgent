# ============================================================
# coordinator/decision.py
# Decision Engine for LiverAI Multi-Agent System
# ============================================================

from typing import Any, Dict, List


class DecisionEngine:
    """
    Global decision engine for the LiverAI multi-agent system.

    The engine:
        - validates agent results
        - evaluates coverage
        - evaluates confidence
        - evaluates trust
        - evaluates quality
        - evaluates conflicts
        - computes risk
        - produces a global decision level
        - keeps heterogeneous tasks separate

    Decision levels:
        HIGH
        MODERATE
        UNCERTAIN
    """

    # ========================================================
    # INITIALIZATION
    # ========================================================

    def __init__(
        self,
        high_confidence: float = 0.80,
        moderate_confidence: float = 0.55,
        high_trust: float = 0.70,
        high_conflict: float = 0.50,
        minimum_coverage: float = 0.50,
        minimum_quality: float = 0.50,
    ):

        self.high_confidence = float(
            high_confidence
        )

        self.moderate_confidence = float(
            moderate_confidence
        )

        self.high_trust = float(
            high_trust
        )

        self.high_conflict = float(
            high_conflict
        )

        self.minimum_coverage = float(
            minimum_coverage
        )

        self.minimum_quality = float(
            minimum_quality
        )

    # ========================================================
    # SAFE FLOAT
    # ========================================================

    @staticmethod
    def _safe_float(
        value: Any,
        default: float = 0.0
    ) -> float:

        try:

            if value is None:
                return default

            value = float(value)

            if value != value:
                return default

            if value == float("inf"):
                return default

            if value == float("-inf"):
                return default

            return value

        except (
            TypeError,
            ValueError
        ):

            return default

    # ========================================================
    # CLIP
    # ========================================================

    @staticmethod
    def _clip01(
        value: Any
    ) -> float:

        value = DecisionEngine._safe_float(
            value,
            default=0.0
        )

        return max(
            0.0,
            min(
                1.0,
                value
            )
        )

    # ========================================================
    # TASK TYPE
    # ========================================================

    @staticmethod
    def _get_task_type(
        result: Dict[str, Any]
    ) -> str:

        return str(
            result.get(
                "task_type",
                result.get(
                    "task",
                    "unknown"
                )
            )
        )

    # ========================================================
    # VALIDATE RESULT
    # ========================================================

    def _is_valid_result(
        self,
        result: Any
    ) -> bool:

        if not isinstance(
            result,
            dict
        ):
            return False

        status = str(
            result.get(
                "status",
                ""
            )
        ).lower()

        if status not in {
            "success",
            "completed"
        }:
            return False

        task_type = self._get_task_type(
            result
        )

        # ----------------------------------------------------
        # SEGMENTATION
        # ----------------------------------------------------

        if task_type == "liver_segmentation":

            details = result.get(
                "details",
                {}
            )

            if not isinstance(
                details,
                dict
            ):
                return False

            has_mask = (
                details.get(
                    "liver_mask"
                ) is not None
            )

            has_probability_map = (
                details.get(
                    "probability_map"
                ) is not None
            )

            return (
                has_mask
                or
                has_probability_map
            )

        # ----------------------------------------------------
        # NORMAL CLASSIFICATION / REASONING
        # ----------------------------------------------------

        return (
            result.get(
                "prediction"
            ) is not None
        )

    # ========================================================
    # CONFLICT SCORE
    # ========================================================

    def _calculate_conflict_score(
        self,
        conflicts: List[Dict[str, Any]]
    ) -> float:

        if not conflicts:
            return 0.0

        values = []

        for conflict in conflicts:

            if not isinstance(
                conflict,
                dict
            ):
                continue

            value = conflict.get(
                "conflict_strength"
            )

            if value is None:

                value = conflict.get(
                    "confidence_gap",
                    0.0
                )

            values.append(
                self._clip01(
                    value
                )
            )

        if not values:
            return 0.0

        return self._clip01(
            sum(values) / len(values)
        )

    # ========================================================
    # GLOBAL PREDICTION
    # ========================================================

    def _extract_global_prediction(
        self,
        valid_results: List[Dict[str, Any]],
        reasoning: Dict[str, Any]
    ):

        # ----------------------------------------------------
        # First priority:
        # reasoning engine prediction
        # ----------------------------------------------------

        if isinstance(
            reasoning,
            dict
        ):

            reasoning_prediction = (
                reasoning.get(
                    "prediction"
                )
            )

            if reasoning_prediction is not None:

                return reasoning_prediction

        # ----------------------------------------------------
        # Do not invent a global prediction from heterogeneous
        # tasks.
        #
        # Example:
        # cirrhosis != tumor != fibrosis != segmentation
        # ----------------------------------------------------

        task_types = set()

        candidates = []

        for result in valid_results:

            task_type = self._get_task_type(
                result
            )

            if task_type == "liver_segmentation":
                continue

            prediction = result.get(
                "prediction"
            )

            if prediction is None:
                continue

            task_types.add(
                task_type
            )

            candidates.append(
                result
            )

        # ----------------------------------------------------
        # Only infer a global prediction if all valid
        # prediction-producing agents belong to the same task.
        # ----------------------------------------------------

        if len(task_types) == 1 and candidates:

            best = max(
                candidates,
                key=lambda r:
                    self._clip01(
                        r.get(
                            "trust",
                            0.0
                        )
                    )
                    *
                    self._clip01(
                        r.get(
                            "confidence",
                            0.0
                        )
                    )
            )

            return best.get(
                "prediction"
            )

        return None

    # ========================================================
    # TASK-LEVEL DECISIONS
    # ========================================================

    def _build_task_decisions(
        self,
        valid_results: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:

        grouped = {}

        for result in valid_results:

            task_type = self._get_task_type(
                result
            )

            grouped.setdefault(
                task_type,
                []
            ).append(
                result
            )

        task_decisions = {}

        for task_type, task_results in grouped.items():

            confidences = [
                self._clip01(
                    r.get(
                        "confidence",
                        0.0
                    )
                )
                for r in task_results
            ]

            trusts = [
                self._clip01(
                    r.get(
                        "trust",
                        0.0
                    )
                )
                for r in task_results
            ]

            qualities = [
                self._clip01(
                    r.get(
                        "quality",
                        0.0
                    )
                )
                for r in task_results
            ]

            mean_confidence = (
                sum(confidences)
                /
                len(confidences)
                if confidences
                else 0.0
            )

            mean_trust = (
                sum(trusts)
                /
                len(trusts)
                if trusts
                else 0.0
            )

            mean_quality = (
                sum(qualities)
                /
                len(qualities)
                if qualities
                else 0.0
            )

            predictions = [
                r.get("prediction")
                for r in task_results
                if r.get("prediction") is not None
            ]

            # ------------------------------------------------
            # Segmentation
            # ------------------------------------------------

            if task_type == "liver_segmentation":

                decision = (
                    "HIGH"
                    if mean_confidence >= self.high_confidence
                    else
                    "MODERATE"
                    if mean_confidence >= self.moderate_confidence
                    else
                    "UNCERTAIN"
                )

                prediction = None

            else:

                prediction = (
                    predictions[0]
                    if predictions
                    else None
                )

                decision = (
                    "HIGH"
                    if (
                        mean_confidence >=
                        self.high_confidence
                        and
                        mean_trust >=
                        self.high_trust
                    )
                    else
                    "MODERATE"
                    if (
                        mean_confidence >=
                        self.moderate_confidence
                    )
                    else
                    "UNCERTAIN"
                )

            task_decisions[
                task_type
            ] = {

                "task_type":
                    task_type,

                "decision":
                    decision,

                "decision_level":
                    decision,

                "prediction":
                    prediction,

                "confidence":
                    mean_confidence,

                "trust":
                    mean_trust,

                "quality":
                    mean_quality,

                "num_agents":
                    len(task_results),

                "request_additional_tests":
                    decision == "UNCERTAIN",
            }

        return task_decisions

    # ========================================================
    # DECIDE
    # ========================================================

    def decide(
        self,
        results,
        conflicts=None,
        reasoning=None,
    ):

        # ----------------------------------------------------
        # Normalize
        # ----------------------------------------------------

        if results is None:
            results = []

        if not isinstance(
            results,
            list
        ):

            results = list(
                results
            )

        conflicts = (
            conflicts
            if isinstance(
                conflicts,
                list
            )
            else []
        )

        reasoning = (
            reasoning
            if isinstance(
                reasoning,
                dict
            )
            else {}
        )

        # ----------------------------------------------------
        # Valid results
        # ----------------------------------------------------

        valid_results = [
            result
            for result in results
            if self._is_valid_result(
                result
            )
        ]

        total_agents = len(
            results
        )

        valid_agents = len(
            valid_results
        )

        coverage = (
            valid_agents /
            total_agents
            if total_agents > 0
            else 0.0
        )

        # ----------------------------------------------------
        # Global metrics
        # ----------------------------------------------------

        confidences = [
            self._clip01(
                r.get(
                    "confidence",
                    0.0
                )
            )
            for r in valid_results
        ]

        trusts = [
            self._clip01(
                r.get(
                    "trust",
                    0.0
                )
            )
            for r in valid_results
        ]

        qualities = [
            self._clip01(
                r.get(
                    "quality",
                    0.0
                )
            )
            for r in valid_results
        ]

        mean_confidence = (
            sum(confidences)
            /
            len(confidences)
            if confidences
            else 0.0
        )

        mean_trust = (
            sum(trusts)
            /
            len(trusts)
            if trusts
            else 0.0
        )

        mean_quality = (
            sum(qualities)
            /
            len(qualities)
            if qualities
            else 0.0
        )

        # ----------------------------------------------------
        # Conflict
        # ----------------------------------------------------

        conflict_score = (
            self._calculate_conflict_score(
                conflicts
            )
        )

        # ----------------------------------------------------
        # Global prediction
        # ----------------------------------------------------

        prediction = (
            self._extract_global_prediction(
                valid_results,
                reasoning
            )
        )

        # ----------------------------------------------------
        # Safety conditions
        # ----------------------------------------------------

        insufficient_data = (
            coverage <
            self.minimum_coverage
            or
            mean_quality <
            self.minimum_quality
        )

        unsafe_conflict = (
            conflict_score >=
            self.high_conflict
        )

        # ----------------------------------------------------
        # Decision level
        # ----------------------------------------------------

        if (
            not valid_results
            or
            insufficient_data
            or
            unsafe_conflict
            or
            mean_confidence <
            self.moderate_confidence
        ):

            decision_level = "UNCERTAIN"

        elif (
            mean_confidence >=
            self.high_confidence
            and
            mean_trust >=
            self.high_trust
            and
            conflict_score < 0.30
        ):

            decision_level = "HIGH"

        else:

            decision_level = "MODERATE"

        # ----------------------------------------------------
        # Risk
        # ----------------------------------------------------

        risk_score = (

            0.40 *
            (
                1.0 -
                mean_confidence
            )

            +

            0.30 *
            (
                1.0 -
                mean_trust
            )

            +

            0.20 *
            conflict_score

            +

            0.10 *
            (
                1.0 -
                mean_quality
            )
        )

        risk_score = self._clip01(
            risk_score
        )

        # ----------------------------------------------------
        # Task-level decisions
        # ----------------------------------------------------

        task_decisions = (
            self._build_task_decisions(
                valid_results
            )
        )

        # ----------------------------------------------------
        # Final result
        # ----------------------------------------------------

        return {

            "status":
                "completed",

            # ------------------------------------------------
            # IMPORTANT:
            # Keep BOTH names for backward compatibility.
            # ------------------------------------------------

            "decision":
                decision_level,

            "decision_level":
                decision_level,

            "prediction":
                prediction,

            "confidence":
                float(
                    mean_confidence
                ),

            "uncertainty":
                float(
                    1.0 -
                    mean_confidence
                ),

            "trust":
                float(
                    mean_trust
                ),

            "quality":
                float(
                    mean_quality
                ),

            "coverage":
                float(
                    coverage
                ),

            "conflict_score":
                float(
                    conflict_score
                ),

            "risk_score":
                float(
                    risk_score
                ),

            "request_additional_tests":
                decision_level ==
                "UNCERTAIN",

            "num_agents":
                total_agents,

            "num_valid_agents":
                valid_agents,

            "task_decisions":
                task_decisions,

            "explanation":
                (
                    f"{valid_agents}/"
                    f"{total_agents} agents "
                    f"provided valid evidence. "
                    f"Mean confidence="
                    f"{mean_confidence:.3f}, "
                    f"mean trust="
                    f"{mean_trust:.3f}, "
                    f"mean quality="
                    f"{mean_quality:.3f}, "
                    f"conflict="
                    f"{conflict_score:.3f}, "
                    f"decision="
                    f"{decision_level}."
                ),
        }


# ============================================================
# BACKWARD COMPATIBILITY
# ============================================================

__all__ = [
    "DecisionEngine"
]
