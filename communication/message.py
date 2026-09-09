# ============================================================
# communication/message.py
# AgentMessage - Communication layer for LiverAI Multi-Agent
# ============================================================

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, Optional


@dataclass
class AgentMessage:
    """
    Standard message exchanged between an AI agent and the Coordinator.

    The message contains:
        - patient information
        - task information
        - prediction/evidence
        - confidence and uncertainty
        - quality and missing-data information
        - trust information
        - communication metadata

    The class is intentionally flexible so it can be used by
    heterogeneous biomedical agents.
    """

    # --------------------------------------------------------
    # IDENTIFICATION
    # --------------------------------------------------------

    patient_id: str = ""

    agent_id: str = ""

    task_type: str = ""

    modality: str = "unknown"

    # --------------------------------------------------------
    # PREDICTION / EVIDENCE
    # --------------------------------------------------------

    prediction: Any = None

    probability: Any = None

    confidence: Optional[float] = None

    uncertainty: Optional[float] = None

    # --------------------------------------------------------
    # INPUT / QUALITY
    # --------------------------------------------------------

    quality: Optional[float] = None

    missing_data_ratio: Optional[float] = None

    # --------------------------------------------------------
    # TRUST
    # --------------------------------------------------------

    trust: Optional[float] = None

    # --------------------------------------------------------
    # COMMUNICATION
    # --------------------------------------------------------

    message_type: str = "PREDICTION"

    request_id: Optional[str] = None

    parent_request_id: Optional[str] = None

    target_agent: Optional[str] = None

    sender: Optional[str] = None

    receiver: Optional[str] = None

    # --------------------------------------------------------
    # ADDITIONAL INFORMATION
    # --------------------------------------------------------

    evidence: Any = None

    details: Dict[str, Any] = field(default_factory=dict)

    metadata: Dict[str, Any] = field(default_factory=dict)

    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    def validate(self) -> Dict[str, Any]:
        """
        Validate the communication message.

        Returns
        -------
        dict
            {
                "valid": bool,
                "errors": list,
                "warnings": list
            }
        """

        errors = []
        warnings = []

        # ----------------------------------------------------
        # REQUIRED IDENTIFIERS
        # ----------------------------------------------------

        if not self.agent_id:
            errors.append("agent_id is required")

        if not self.task_type:
            errors.append("task_type is required")

        # patient_id may be optional for some technical tasks,
        # therefore it generates a warning instead of an error.
        if not self.patient_id:
            warnings.append("patient_id is missing")

        # ----------------------------------------------------
        # CONFIDENCE
        # ----------------------------------------------------

        if self.confidence is not None:

            try:
                confidence = float(self.confidence)

                if not 0.0 <= confidence <= 1.0:
                    errors.append(
                        "confidence must be between 0 and 1"
                    )

            except (TypeError, ValueError):

                errors.append(
                    "confidence must be numeric"
                )

        # ----------------------------------------------------
        # UNCERTAINTY
        # ----------------------------------------------------

        if self.uncertainty is not None:

            try:
                uncertainty = float(self.uncertainty)

                if not 0.0 <= uncertainty <= 1.0:
                    errors.append(
                        "uncertainty must be between 0 and 1"
                    )

            except (TypeError, ValueError):

                errors.append(
                    "uncertainty must be numeric"
                )

        # ----------------------------------------------------
        # QUALITY
        # ----------------------------------------------------

        if self.quality is not None:

            try:
                quality = float(self.quality)

                if not 0.0 <= quality <= 1.0:
                    errors.append(
                        "quality must be between 0 and 1"
                    )

            except (TypeError, ValueError):

                errors.append(
                    "quality must be numeric"
                )

        # ----------------------------------------------------
        # MISSING DATA
        # ----------------------------------------------------

        if self.missing_data_ratio is not None:

            try:
                missing_ratio = float(
                    self.missing_data_ratio
                )

                if not 0.0 <= missing_ratio <= 1.0:
                    errors.append(
                        "missing_data_ratio must be between 0 and 1"
                    )

            except (TypeError, ValueError):

                errors.append(
                    "missing_data_ratio must be numeric"
                )

        # ----------------------------------------------------
        # TRUST
        # ----------------------------------------------------

        if self.trust is not None:

            try:
                trust = float(self.trust)

                if not 0.0 <= trust <= 1.0:
                    errors.append(
                        "trust must be between 0 and 1"
                    )

            except (TypeError, ValueError):

                errors.append(
                    "trust must be numeric"
                )

        # ----------------------------------------------------
        # MESSAGE TYPE
        # ----------------------------------------------------

        allowed_message_types = {
            "PREDICTION",
            "REQUEST_PREDICTION",
            "REQUEST_REASSESSMENT",
            "REQUEST_EXPLANATION",
            "REQUEST_SEGMENTATION",
            "REQUEST_ADDITIONAL_EVIDENCE",
            "RESPONSE",
            "FEEDBACK",
            "ALERT",
            "ERROR",
        }

        if self.message_type not in allowed_message_types:

            warnings.append(
                f"Unknown message_type: {self.message_type}"
            )

        # ----------------------------------------------------
        # FINAL VALIDATION
        # ----------------------------------------------------

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
        }

    # --------------------------------------------------------
    # SERIALIZATION
    # --------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the message into a dictionary.
        """

        return asdict(self)

    # --------------------------------------------------------
    # ALIAS
    # --------------------------------------------------------

    def dict(self) -> Dict[str, Any]:
        """
        Compatibility alias for to_dict().
        """

        return self.to_dict()

    # --------------------------------------------------------
    # FACTORY FROM DICTIONARY
    # --------------------------------------------------------

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any]
    ) -> "AgentMessage":
        """
        Create an AgentMessage from a dictionary.
        """

        if not isinstance(data, dict):
            raise TypeError(
                "AgentMessage.from_dict expects a dictionary"
            )

        # Keep only fields supported by the dataclass.
        valid_fields = {
            "patient_id",
            "agent_id",
            "task_type",
            "modality",
            "prediction",
            "probability",
            "confidence",
            "uncertainty",
            "quality",
            "missing_data_ratio",
            "trust",
            "message_type",
            "request_id",
            "parent_request_id",
            "target_agent",
            "sender",
            "receiver",
            "evidence",
            "details",
            "metadata",
            "timestamp",
        }

        filtered_data = {
            key: value
            for key, value in data.items()
            if key in valid_fields
        }

        return cls(**filtered_data)

    # --------------------------------------------------------
    # REQUEST FACTORY
    # --------------------------------------------------------

    @classmethod
    def create_request(
        cls,
        patient_id: str,
        sender: str,
        receiver: Optional[str],
        task_type: str,
        message_type: str,
        request_id: Optional[str] = None,
        parent_request_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "AgentMessage":
        """
        Create a Coordinator -> Agent request.
        """

        return cls(
            patient_id=patient_id,
            agent_id=sender,
            task_type=task_type,
            message_type=message_type,
            request_id=request_id,
            parent_request_id=parent_request_id,
            sender=sender,
            receiver=receiver,
            details=details or {},
            metadata=metadata or {},
        )

    # --------------------------------------------------------
    # RESPONSE FACTORY
    # --------------------------------------------------------

    @classmethod
    def create_response(
        cls,
        patient_id: str,
        agent_id: str,
        task_type: str,
        prediction: Any = None,
        probability: Any = None,
        confidence: Optional[float] = None,
        uncertainty: Optional[float] = None,
        quality: Optional[float] = None,
        missing_data_ratio: Optional[float] = None,
        trust: Optional[float] = None,
        modality: str = "unknown",
        evidence: Any = None,
        details: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
        parent_request_id: Optional[str] = None,
    ) -> "AgentMessage":
        """
        Create an Agent -> Coordinator response.
        """

        return cls(
            patient_id=patient_id,
            agent_id=agent_id,
            task_type=task_type,
            modality=modality,
            prediction=prediction,
            probability=probability,
            confidence=confidence,
            uncertainty=uncertainty,
            quality=quality,
            missing_data_ratio=missing_data_ratio,
            trust=trust,
            message_type="RESPONSE",
            request_id=request_id,
            parent_request_id=parent_request_id,
            sender=agent_id,
            receiver="Coordinator",
            evidence=evidence,
            details=details or {},
        )

    # --------------------------------------------------------
    # FEEDBACK FACTORY
    # --------------------------------------------------------

    @classmethod
    def create_feedback(
        cls,
        patient_id: str,
        sender: str,
        receiver: Optional[str],
        task_type: str,
        details: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "AgentMessage":
        """
        Create an agent/coordinator feedback message.
        """

        return cls(
            patient_id=patient_id,
            agent_id=sender,
            task_type=task_type,
            message_type="FEEDBACK",
            sender=sender,
            receiver=receiver,
            details=details or {},
            metadata=metadata or {},
        )

    # --------------------------------------------------------
    # STRING REPRESENTATION
    # --------------------------------------------------------

    def __str__(self) -> str:

        return (
            f"AgentMessage("
            f"agent_id={self.agent_id}, "
            f"task_type={self.task_type}, "
            f"message_type={self.message_type}, "
            f"prediction={self.prediction}, "
            f"confidence={self.confidence}"
            f")"
        )

    def __repr__(self) -> str:

        return self.__str__()
