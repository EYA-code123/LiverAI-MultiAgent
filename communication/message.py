# =============================================================================
# LiverAI-MultiAgent
# COMMUNICATION / MESSAGE
# =============================================================================
#
# Standardized message exchanged between the Coordinator and specialist agents.
#
# Supported requests:
#   - REQUEST_PREDICTION
#   - REQUEST_REASSESSMENT
#   - REQUEST_EXPLANATION
#   - REQUEST_SEGMENTATION
#   - REQUEST_ADDITIONAL_EVIDENCE
#
# =============================================================================

from datetime import datetime
from uuid import uuid4
from typing import Any, Dict, Optional


class AgentMessage:
    """
    Structured communication message used by LiverAI-MultiAgent.

    The message is intentionally model-independent. It transports:
        - sender / receiver
        - patient
        - medical task
        - request type
        - input/output payload
        - correlation information
    """

    # -------------------------------------------------------------------------
    # REQUEST TYPES
    # -------------------------------------------------------------------------

    REQUEST_PREDICTION = "REQUEST_PREDICTION"

    REQUEST_REASSESSMENT = "REQUEST_REASSESSMENT"

    REQUEST_EXPLANATION = "REQUEST_EXPLANATION"

    REQUEST_SEGMENTATION = "REQUEST_SEGMENTATION"

    REQUEST_ADDITIONAL_EVIDENCE = (
        "REQUEST_ADDITIONAL_EVIDENCE"
    )

    # -------------------------------------------------------------------------
    # MESSAGE TYPES
    # -------------------------------------------------------------------------

    TYPE_REQUEST = "request"

    TYPE_RESPONSE = "response"

    TYPE_FEEDBACK = "feedback"

    TYPE_NOTIFICATION = "notification"

    # -------------------------------------------------------------------------
    # INITIALIZATION
    # -------------------------------------------------------------------------

    def __init__(
        self,
        sender: Optional[str] = None,
        receiver: Optional[str] = None,
        message_type: str = TYPE_REQUEST,
        request_type: Optional[str] = None,
        task_type: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
        patient_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
        message_id: Optional[str] = None,
        timestamp: Optional[str] = None,

        # Compatibility aliases used by some callers
        agent_id: Optional[str] = None,
        data: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):

        # ---------------------------------------------------------------------
        # Compatibility
        # ---------------------------------------------------------------------

        if sender is None and agent_id is not None:
            sender = agent_id

        if payload is None and data is not None:
            payload = data

        self.message_id = (
            str(message_id)
            if message_id is not None
            else str(uuid4())
        )

        self.correlation_id = (
            str(correlation_id)
            if correlation_id is not None
            else str(uuid4())
        )

        self.sender = sender

        self.receiver = receiver

        self.message_type = (
            message_type
            if message_type is not None
            else self.TYPE_REQUEST
        )

        self.request_type = request_type

        self.task_type = task_type

        self.patient_id = patient_id

        self.payload = (
            dict(payload)
            if isinstance(payload, dict)
            else {}
        )

        self.metadata = (
            dict(metadata)
            if isinstance(metadata, dict)
            else {}
        )

        self.timestamp = (
            timestamp
            if timestamp is not None
            else datetime.now().isoformat()
        )

    # =========================================================================
    # VALIDATION
    # =========================================================================

    def validate(self) -> Dict[str, Any]:
        """
        Validate the structure of the message.

        Returns:
            {
                "valid": bool,
                "errors": [...]
            }
        """

        errors = []

        if not self.sender:
            errors.append(
                "sender is required"
            )

        if not self.receiver:
            errors.append(
                "receiver is required"
            )

        valid_message_types = {
            self.TYPE_REQUEST,
            self.TYPE_RESPONSE,
            self.TYPE_FEEDBACK,
            self.TYPE_NOTIFICATION,
        }

        if self.message_type not in valid_message_types:
            errors.append(
                f"invalid message_type: "
                f"{self.message_type}"
            )

        valid_request_types = {
            self.REQUEST_PREDICTION,
            self.REQUEST_REASSESSMENT,
            self.REQUEST_EXPLANATION,
            self.REQUEST_SEGMENTATION,
            self.REQUEST_ADDITIONAL_EVIDENCE,
            None,
        }

        if self.request_type not in valid_request_types:
            errors.append(
                f"invalid request_type: "
                f"{self.request_type}"
            )

        if not self.correlation_id:
            errors.append(
                "correlation_id is required"
            )

        if not isinstance(
            self.payload,
            dict
        ):
            errors.append(
                "payload must be a dictionary"
            )

        return {
            "valid": len(errors) == 0,
            "errors": errors,
        }

    # =========================================================================
    # SERIALIZATION
    # =========================================================================

    def to_dict(self) -> Dict[str, Any]:

        return {
            "message_id": self.message_id,

            "correlation_id":
                self.correlation_id,

            "sender":
                self.sender,

            "receiver":
                self.receiver,

            "message_type":
                self.message_type,

            "request_type":
                self.request_type,

            "task_type":
                self.task_type,

            "patient_id":
                self.patient_id,

            "payload":
                self.payload,

            "metadata":
                self.metadata,

            "timestamp":
                self.timestamp,
        }

    # =========================================================================
    # DESERIALIZATION
    # =========================================================================

    @classmethod
    def from_dict(
        cls,
        data: Optional[Dict[str, Any]]
    ):

        if not isinstance(
            data,
            dict
        ):
            raise TypeError(
                "AgentMessage.from_dict expects a dictionary."
            )

        return cls(

            sender=data.get(
                "sender"
            ),

            receiver=data.get(
                "receiver"
            ),

            message_type=data.get(
                "message_type",
                cls.TYPE_REQUEST
            ),

            request_type=data.get(
                "request_type"
            ),

            task_type=data.get(
                "task_type"
            ),

            payload=data.get(
                "payload",
                {}
            ),

            patient_id=data.get(
                "patient_id"
            ),

            correlation_id=data.get(
                "correlation_id"
            ),

            message_id=data.get(
                "message_id"
            ),

            timestamp=data.get(
                "timestamp"
            ),

            metadata=data.get(
                "metadata",
                {}
            ),
        )

    # =========================================================================
    # REPRESENTATION
    # =========================================================================

    def __repr__(self):

        return (
            "AgentMessage("
            f"sender={self.sender!r}, "
            f"receiver={self.receiver!r}, "
            f"message_type={self.message_type!r}, "
            f"request_type={self.request_type!r}, "
            f"task_type={self.task_type!r}, "
            f"patient_id={self.patient_id!r}"
            ")"
        )


# =============================================================================
# COMMUNICATION PROTOCOL
# =============================================================================


class CommunicationProtocol:
    """
    Communication manager used by LiverCoordinator.

    It stores a structured communication trace so that the orchestration
    process remains observable and auditable.
    """

    def __init__(
        self,
        coordinator_id="LiverCoordinator"
    ):

        self.coordinator_id = coordinator_id

        self.messages = []

    # =========================================================================
    # REQUEST
    # =========================================================================

    def create_request(
        self,
        receiver,
        request_type,
        task_type=None,
        payload=None,
        patient_id=None,
        metadata=None,
    ):

        message = AgentMessage(

            sender=self.coordinator_id,

            receiver=receiver,

            message_type=AgentMessage.TYPE_REQUEST,

            request_type=request_type,

            task_type=task_type,

            payload=payload,

            patient_id=patient_id,

            metadata=metadata,
        )

        validation = message.validate()

        if not validation["valid"]:
            raise ValueError(
                validation["errors"]
            )

        self.messages.append(
            message
        )

        return message

    # =========================================================================
    # RESPONSE
    # =========================================================================

    def create_response(
        self,
        receiver,
        request_message,
        payload=None,
        task_type=None,
        patient_id=None,
        metadata=None,
    ):

        if not isinstance(
            request_message,
            AgentMessage
        ):
            raise TypeError(
                "request_message must be AgentMessage."
            )

        message = AgentMessage(

            sender=self.coordinator_id,

            receiver=receiver,

            message_type=AgentMessage.TYPE_RESPONSE,

            request_type=request_message.request_type,

            task_type=(
                task_type
                if task_type is not None
                else request_message.task_type
            ),

            payload=payload,

            patient_id=(
                patient_id
                if patient_id is not None
                else request_message.patient_id
            ),

            correlation_id=
                request_message.correlation_id,

            metadata=metadata,
        )

        self.messages.append(
            message
        )

        return message

    # =========================================================================
    # AGENT RESPONSE
    # =========================================================================

    def record_agent_response(
        self,
        sender,
        request_message,
        payload=None,
        task_type=None,
        patient_id=None,
        metadata=None,
    ):

        if not isinstance(
            request_message,
            AgentMessage
        ):
            raise TypeError(
                "request_message must be AgentMessage."
            )

        message = AgentMessage(

            sender=sender,

            receiver=self.coordinator_id,

            message_type=AgentMessage.TYPE_RESPONSE,

            request_type=request_message.request_type,

            task_type=(
                task_type
                if task_type is not None
                else request_message.task_type
            ),

            payload=payload,

            patient_id=(
                patient_id
                if patient_id is not None
                else request_message.patient_id
            ),

            correlation_id=
                request_message.correlation_id,

            metadata=metadata,
        )

        self.messages.append(
            message
        )

        return message

    # =========================================================================
    # FEEDBACK
    # =========================================================================

    def create_feedback(
        self,
        receiver,
        task_type=None,
        payload=None,
        patient_id=None,
        correlation_id=None,
    ):

        message = AgentMessage(

            sender=self.coordinator_id,

            receiver=receiver,

            message_type=AgentMessage.TYPE_FEEDBACK,

            request_type=None,

            task_type=task_type,

            payload=payload,

            patient_id=patient_id,

            correlation_id=correlation_id,
        )

        self.messages.append(
            message
        )

        return message

    # =========================================================================
    # TRACE
    # =========================================================================

    def get_trace(self):

        return [
            message.to_dict()
            for message in self.messages
        ]

    # =========================================================================
    # SUMMARY
    # =========================================================================

    def summary(self):

        requests = sum(
            1
            for message in self.messages
            if message.message_type
            ==
            AgentMessage.TYPE_REQUEST
        )

        responses = sum(
            1
            for message in self.messages
            if message.message_type
            ==
            AgentMessage.TYPE_RESPONSE
        )

        feedback = sum(
            1
            for message in self.messages
            if message.message_type
            ==
            AgentMessage.TYPE_FEEDBACK
        )

        return {

            "total_messages":
                len(self.messages),

            "requests":
                requests,

            "responses":
                responses,

            "feedback":
                feedback,

            "trace":
                self.get_trace(),
        }

    # =========================================================================
    # CLEAR
    # =========================================================================

    def clear(self):

        self.messages.clear()


__all__ = [
    "AgentMessage",
    "CommunicationProtocol",
]
