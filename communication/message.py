# communication/message.py
# ============================================================
# Agent Communication Message
# LiverAI Multi-Agent System
# ============================================================

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, Optional
import uuid


# ============================================================
# SUPPORTED MESSAGE TYPES
# ============================================================

REQUEST_PREDICTION = "REQUEST_PREDICTION"
REQUEST_REASSESSMENT = "REQUEST_REASSESSMENT"
REQUEST_EXPLANATION = "REQUEST_EXPLANATION"
REQUEST_SEGMENTATION = "REQUEST_SEGMENTATION"
REQUEST_ADDITIONAL_EVIDENCE = "REQUEST_ADDITIONAL_EVIDENCE"

RESPONSE_PREDICTION = "RESPONSE_PREDICTION"
RESPONSE_REASSESSMENT = "RESPONSE_REASSESSMENT"
RESPONSE_EXPLANATION = "RESPONSE_EXPLANATION"
RESPONSE_SEGMENTATION = "RESPONSE_SEGMENTATION"
RESPONSE_ADDITIONAL_EVIDENCE = "RESPONSE_ADDITIONAL_EVIDENCE"

FEEDBACK = "FEEDBACK"

SUPPORTED_MESSAGE_TYPES = {
    REQUEST_PREDICTION,
    REQUEST_REASSESSMENT,
    REQUEST_EXPLANATION,
    REQUEST_SEGMENTATION,
    REQUEST_ADDITIONAL_EVIDENCE,
    RESPONSE_PREDICTION,
    RESPONSE_REASSESSMENT,
    RESPONSE_EXPLANATION,
    RESPONSE_SEGMENTATION,
    RESPONSE_ADDITIONAL_EVIDENCE,
    FEEDBACK,
}


# ============================================================
# AGENT MESSAGE
# ============================================================

@dataclass
class AgentMessage:

    patient_id: Optional[str] = None

    agent_id: Optional[str] = None
    task_type: Optional[str] = None
    modality: str = "unknown"

    prediction: Any = None
    probability: Any = None

    confidence: float = 0.0
    uncertainty: float = 1.0
    quality: float = 1.0
    missing_data_ratio: float = 0.0
    trust: float = 0.0

    message_type: str = RESPONSE_PREDICTION

    request_id: str = field(
        default_factory=lambda: str(uuid.uuid4())
    )

    parent_request_id: Optional[str] = None

    target_agent: Optional[str] = None

    sender: Optional[str] = None
    receiver: Optional[str] = None

    evidence: Dict[str, Any] = field(default_factory=dict)
    details: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    # ========================================================
    # VALIDATION
    # ========================================================

    def validate(self) -> Dict[str, Any]:

        errors = []

        if not self.agent_id:
            errors.append("agent_id is required")

        if not self.task_type:
            errors.append("task_type is required")

        if self.message_type not in SUPPORTED_MESSAGE_TYPES:
            errors.append(
                f"Unsupported message_type: {self.message_type}"
            )

        try:
            confidence = float(self.confidence)

            if not 0.0 <= confidence <= 1.0:
                errors.append(
                    "confidence must be between 0 and 1"
                )

        except Exception:
            errors.append("confidence must be numeric")

        try:
            uncertainty = float(self.uncertainty)

            if not 0.0 <= uncertainty <= 1.0:
                errors.append(
                    "uncertainty must be between 0 and 1"
                )

        except Exception:
            errors.append("uncertainty must be numeric")

        try:
            quality = float(self.quality)

            if not 0.0 <= quality <= 1.0:
                errors.append(
                    "quality must be between 0 and 1"
                )

        except Exception:
            errors.append("quality must be numeric")

        try:
            missing_ratio = float(self.missing_data_ratio)

            if not 0.0 <= missing_ratio <= 1.0:
                errors.append(
                    "missing_data_ratio must be between 0 and 1"
                )

        except Exception:
            errors.append(
                "missing_data_ratio must be numeric"
            )

        try:
            trust = float(self.trust)

            if not 0.0 <= trust <= 1.0:
                errors.append(
                    "trust must be between 0 and 1"
                )

        except Exception:
            errors.append("trust must be numeric")

        return {
            "valid": len(errors) == 0,
            "errors": errors,
        }

    # ========================================================
    # SERIALIZATION
    # ========================================================

    def to_dict(self) -> Dict[str, Any]:

        data = asdict(self)

        return data

    # Compatibility alias
    def dict(self) -> Dict[str, Any]:
        return self.to_dict()

    # ========================================================
    # DESERIALIZATION
    # ========================================================

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any]
    ) -> "AgentMessage":

        if not isinstance(data, dict):
            raise TypeError(
                "AgentMessage.from_dict expects a dictionary"
            )

        allowed_fields = {
            field_name
            for field_name in cls.__dataclass_fields__
        }

        clean_data = {
            key: value
            for key, value in data.items()
            if key in allowed_fields
        }

        return cls(**clean_data)

    # ========================================================
    # REQUEST FACTORY
    # ========================================================

    @classmethod
    def create_request(
        cls,
        patient_id: Optional[str],
        sender: str,
        receiver: Optional[str],
        task_type: str,
        message_type: str,
        modality: str = "unknown",
        details: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        parent_request_id: Optional[str] = None,
    ) -> "AgentMessage":

        return cls(
            patient_id=patient_id,
            agent_id=sender,
            task_type=task_type,
            modality=modality,
            message_type=message_type,
            sender=sender,
            receiver=receiver,
            target_agent=receiver,
            details=details or {},
            metadata=metadata or {},
            parent_request_id=parent_request_id,
        )

    # ========================================================
    # RESPONSE FACTORY
    # ========================================================

    @classmethod
    def create_response(
        cls,
        patient_id: Optional[str],
        sender: str,
        receiver: Optional[str],
        task_type: str,
        prediction: Any = None,
        probability: Any = None,
        confidence: float = 0.0,
        uncertainty: float = 1.0,
        quality: float = 1.0,
        missing_data_ratio: float = 0.0,
        trust: float = 0.0,
        modality: str = "unknown",
        message_type: str = RESPONSE_PREDICTION,
        evidence: Optional[Dict[str, Any]] = None,
        details: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        parent_request_id: Optional[str] = None,
    ) -> "AgentMessage":

        return cls(
            patient_id=patient_id,
            agent_id=sender,
            task_type=task_type,
            modality=modality,
            prediction=prediction,
            probability=probability,
            confidence=confidence,
            uncertainty=uncertainty,
            quality=quality,
            missing_data_ratio=missing_data_ratio,
            trust=trust,
            message_type=message_type,
            parent_request_id=parent_request_id,
            sender=sender,
            receiver=receiver,
            target_agent=receiver,
            evidence=evidence or {},
            details=details or {},
            metadata=metadata or {},
        )

    # ========================================================
    # FEEDBACK FACTORY
    # ========================================================

    @classmethod
    def create_feedback(
        cls,
        patient_id: Optional[str],
        sender: str,
        receiver: Optional[str],
        task_type: str,
        details: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        parent_request_id: Optional[str] = None,
    ) -> "AgentMessage":

        return cls(
            patient_id=patient_id,
            agent_id=sender,
            task_type=task_type,
            message_type=FEEDBACK,
            sender=sender,
            receiver=receiver,
            target_agent=receiver,
            details=details or {},
            metadata=metadata or {},
            parent_request_id=parent_request_id,
        )

    # ========================================================
    # REPRESENTATION
    # ========================================================

    def __str__(self) -> str:

        return (
            f"AgentMessage("
            f"agent={self.agent_id}, "
            f"task={self.task_type}, "
            f"type={self.message_type}, "
            f"prediction={self.prediction}, "
            f"confidence={self.confidence:.3f}"
            f")"
        )

    def __repr__(self) -> str:

        return self.__str__()


# ============================================================
# COMMUNICATION PROTOCOL
# ============================================================

class CommunicationProtocol:
    """
    Compatibility communication layer used by the Coordinator.

    It provides a lightweight protocol for creating, validating,
    sending and recording AgentMessage objects.

    This class is intentionally independent from the actual
    transport mechanism. It can therefore be used for:
        - local orchestration
        - in-process communication
        - future asynchronous communication
        - agent-to-agent delegation
    """

    def __init__(
        self,
        coordinator: Any = None,
        **kwargs: Any
    ):

        self.coordinator = coordinator

        self.messages = []
        self.history = self.messages

        self.requests = []
        self.responses = []
        self.feedback = []

    # ========================================================
    # SEND
    # ========================================================

    def send(
        self,
        message: Any
    ) -> Dict[str, Any]:

        if isinstance(message, dict):
            message = AgentMessage.from_dict(message)

        if not isinstance(message, AgentMessage):
            raise TypeError(
                "message must be an AgentMessage or dictionary"
            )

        validation = message.validate()

        if not validation["valid"]:
            return {
                "status": "error",
                "success": False,
                "message": message.to_dict(),
                "validation": validation,
                "error": "Invalid AgentMessage",
            }

        self.messages.append(message)

        if message.message_type.startswith("REQUEST_"):
            self.requests.append(message)

        elif message.message_type.startswith("RESPONSE_"):
            self.responses.append(message)

        elif message.message_type == FEEDBACK:
            self.feedback.append(message)

        return {
            "status": "success",
            "success": True,
            "message": message.to_dict(),
        }

    # ========================================================
    # PUBLISH ALIAS
    # ========================================================

    def publish(
        self,
        message: Any
    ) -> Dict[str, Any]:

        return self.send(message)

    # ========================================================
    # SEND REQUEST
    # ========================================================

    def send_request(
        self,
        sender: str,
        receiver: Optional[str],
        task_type: str,
        message_type: str = REQUEST_PREDICTION,
        patient_id: Optional[str] = None,
        modality: str = "unknown",
        details: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        parent_request_id: Optional[str] = None,
    ) -> Dict[str, Any]:

        message = AgentMessage.create_request(
            patient_id=patient_id,
            sender=sender,
            receiver=receiver,
            task_type=task_type,
            message_type=message_type,
            modality=modality,
            details=details,
            metadata=metadata,
            parent_request_id=parent_request_id,
        )

        return self.send(message)

    # ========================================================
    # RECORD RESPONSE
    # ========================================================

    def record_response(
        self,
        message: Any
    ) -> Dict[str, Any]:

        return self.send(message)

    # ========================================================
    # RECORD FEEDBACK
    # ========================================================

    def record_feedback(
        self,
        message: Any
    ) -> Dict[str, Any]:

        return self.send(message)

    # ========================================================
    # HISTORY
    # ========================================================

    def get_history(self):

        return list(self.messages)

    # Compatibility aliases
    get_messages = get_history
    history_messages = get_history

    # ========================================================
    # CLEAR
    # ========================================================

    def clear(self):

        self.messages.clear()
        self.requests.clear()
        self.responses.clear()
        self.feedback.clear()


# ============================================================
# MODULE EXPORTS
# ============================================================

__all__ = [
    "AgentMessage",
    "CommunicationProtocol",

    "REQUEST_PREDICTION",
    "REQUEST_REASSESSMENT",
    "REQUEST_EXPLANATION",
    "REQUEST_SEGMENTATION",
    "REQUEST_ADDITIONAL_EVIDENCE",

    "RESPONSE_PREDICTION",
    "RESPONSE_REASSESSMENT",
    "RESPONSE_EXPLANATION",
    "RESPONSE_SEGMENTATION",
    "RESPONSE_ADDITIONAL_EVIDENCE",

    "FEEDBACK",

    "SUPPORTED_MESSAGE_TYPES",
]
