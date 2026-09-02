"""
Agent Adapter
=============

Converts heterogeneous specialist-agent outputs into a unified
Adaptive Coordination Intelligence representation.

This module is deliberately independent from the individual
medical models so existing agents do not need to be rewritten.
"""

import time
import math
import numpy as np

from communication.message import AgentMessage


class AgentAdapter:

    TASK_TYPES = {
        "cirrhosis": "classification",
        "fatty_liver": "classification",
        "fibrosis": "regression",
        "tumor": "classification",
        "tumor_classification": "classification",
        "segmentation": "segmentation",
        "liver_segmentation": "segmentation",
        "clinical_reasoning": "reasoning",
    }

    MODALITIES = {
        "cirrhosis": "clinical",
        "fatty_liver": "clinical",
        "fibrosis": "clinical",
        "tumor": "medical_image",
        "tumor_classification": "medical_image",
        "segmentation": "medical_image",
        "liver_segmentation": "medical_image",
        "clinical_reasoning": "multimodal",
    }

    def __init__(self):
        pass

    @staticmethod
    def _clip(value, low=0.0, high=1.0):
        try:
            return float(np.clip(float(value), low, high))
        except Exception:
            return low

    def compute_missing_ratio(self, data):

        if data is None:
            return 1.0

        try:

            if hasattr(data, "isnull"):
                total = data.size

                if total == 0:
                    return 0.0

                return float(data.isnull().sum().sum() / total)

            if isinstance(data, dict):

                if len(data) == 0:
                    return 0.0

                missing = 0

                for value in data.values():

                    if value is None:
                        missing += 1

                    elif isinstance(value, float) and math.isnan(value):
                        missing += 1

                return missing / len(data)

            if isinstance(data, (list, tuple, np.ndarray)):

                arr = np.asarray(data, dtype=object)

                if arr.size == 0:
                    return 0.0

                missing = 0

                for value in arr.flatten():

                    if value is None:
                        missing += 1

                    else:
                        try:
                            if np.isnan(value):
                                missing += 1
                        except Exception:
                            pass

                return missing / arr.size

        except Exception:
            pass

        return 0.0

    def compute_data_quality(
        self,
        data,
        missing_ratio=None
    ):

        if missing_ratio is None:
            missing_ratio = self.compute_missing_ratio(data)

        quality = 1.0 - missing_ratio

        return self._clip(quality)

    def extract_probabilities(self, result):

        probabilities = {}

        if not isinstance(result, dict):
            return probabilities

        # Existing class_probabilities
        cp = result.get("class_probabilities")

        if isinstance(cp, dict):

            for key, value in cp.items():

                try:
                    probabilities[str(key)] = float(value)
                except Exception:
                    pass

        # probability may already be a dictionary
        probability = result.get("probability")

        if isinstance(probability, dict):

            for key, value in probability.items():

                try:
                    probabilities[str(key)] = float(value)
                except Exception:
                    pass

        # Normalize
        if probabilities:

            total = sum(
                max(0.0, value)
                for value in probabilities.values()
            )

            if total > 0:

                probabilities = {
                    key: value / total
                    for key, value in probabilities.items()
                }

        return probabilities

    def compute_confidence(
        self,
        result,
        probabilities=None
    ):

        if probabilities:

            return self._clip(
                max(probabilities.values())
            )

        value = result.get("confidence")

        if value is not None:
            return self._clip(value)

        probability = result.get("probability")

        if isinstance(probability, (float, int)):

            return self._clip(probability)

        return 0.5

    def compute_uncertainty(
        self,
        result,
        confidence
    ):

        value = result.get("uncertainty")

        if value is not None:

            return self._clip(value)

        return self._clip(1.0 - confidence)

    def extract_feature_importance(
        self,
        result,
        agent=None
    ):

        if isinstance(result, dict):

            importance = result.get(
                "feature_importance"
            )

            if isinstance(importance, dict):
                return importance

            details = result.get("details", {})

            if isinstance(details, dict):

                importance = details.get(
                    "feature_importance"
                )

                if isinstance(importance, dict):
                    return importance

        # Try sklearn-like model
        if agent is not None:

            model = getattr(agent, "model", None)

            if model is not None:

                try:

                    values = getattr(
                        model,
                        "feature_importances_"
                    )

                    features = getattr(
                        model,
                        "feature_names_in_",
                        None
                    )

                    if features is not None:

                        return {
                            str(feature): float(value)
                            for feature, value
                            in zip(features, values)
                        }

                except Exception:
                    pass

        return {}

    def adapt(
        self,
        result,
        patient_id=None,
        agent=None,
        input_data=None,
        latency_ms=None
    ):

        if isinstance(result, AgentMessage):
            return result

        if result is None:

            return AgentMessage(
                patient_id=patient_id,
                agent_id="unknown",
                status="error",
                error="Agent returned None."
            )

        if not isinstance(result, dict):

            result = {
                "prediction": result
            }

        agent_id = (
            result.get("agent_id")
            or result.get("agent")
            or getattr(agent, "agent_id", None)
            or "unknown_agent"
        )

        normalized_id = str(agent_id).lower()

        # Identify task
        task_type = result.get("task_type")

        details = result.get("details", {})

        if not task_type and isinstance(details, dict):
            task_type = details.get("task_type")

        if not task_type:
            task_type = self.TASK_TYPES.get(
                normalized_id,
                "unknown"
            )

        # Identify modality
        modality = result.get("modality")

        if not modality:
            modality = self.MODALITIES.get(
                normalized_id,
                "unknown"
            )

        probabilities = self.extract_probabilities(result)

        confidence = self.compute_confidence(
            result,
            probabilities
        )

        uncertainty = self.compute_uncertainty(
            result,
            confidence
        )

        missing_ratio = self.compute_missing_ratio(
            input_data
        )

        data_quality = self.compute_data_quality(
            input_data,
            missing_ratio
        )

        # Preserve existing quality if meaningful
        existing_quality = result.get("quality")

        if existing_quality is not None:

            quality = self._clip(
                existing_quality
            )

            # Prevent constant quality=1 from hiding
            # poor input data.
            quality = min(
                quality,
                data_quality
            )

        else:
            quality = data_quality

        feature_importance = (
            self.extract_feature_importance(
                result,
                agent
            )
        )

        prediction = result.get(
            "prediction"
        )

        error = result.get("error")

        status = result.get(
            "status",
            "success"
        )

        if error:
            status = "error"

        if latency_ms is None:
            latency_ms = result.get(
                "latency_ms",
                0.0
            )

        explanation = result.get(
            "explanation",
            ""
        )

        embeddings = result.get(
            "embeddings"
        )

        return AgentMessage(
            patient_id=patient_id,
            agent_id=str(agent_id),
            prediction=prediction,
            probability=result.get(
                "probability"
            ),
            confidence=confidence,
            uncertainty=uncertainty,
            quality=quality,
            missing_data_ratio=missing_ratio,
            trust=0.5,
            utility=0.0,
            agreement=0.0,
            stability=0.0,
            task_type=str(task_type),
            modality=str(modality),
            latency_ms=float(latency_ms or 0.0),
            class_probabilities=probabilities,
            feature_importance=feature_importance,
            explanation=explanation,
            embeddings=embeddings,
            details=details,
            status=status,
            error=error,
        )

    def adapt_many(
        self,
        results,
        patient_id=None,
        agents=None,
        input_data=None
    ):

        messages = []

        agents = agents or {}

        for agent_id, result in results.items():

            agent = agents.get(agent_id)

            start = time.perf_counter()

            message = self.adapt(
                result=result,
                patient_id=patient_id,
                agent=agent,
                input_data=input_data
            )

            if message.latency_ms == 0:

                message.latency_ms = (
                    time.perf_counter() - start
                ) * 1000.0

            messages.append(message)

        return messages
