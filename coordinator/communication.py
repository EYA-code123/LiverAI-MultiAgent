# ============================================================
# LIVERAI - AGENT COMMUNICATION PROTOCOL
# ============================================================

from datetime import datetime
from uuid import uuid4


class AgentMessage:
    """
    Standardized communication message between
    LiverCoordinator and specialist agents.
    """

    # --------------------------------------------------------
    # REQUEST TYPES REQUIRED BY THE STUDENT GUIDE
    # --------------------------------------------------------

    REQUEST_PREDICTION = "REQUEST_PREDICTION"
    REQUEST_REASSESSMENT = "REQUEST_REASSESSMENT"
    REQUEST_EXPLANATION = "REQUEST_EXPLANATION"
    REQUEST_SEGMENTATION = "REQUEST_SEGMENTATION"
    REQUEST_ADDITIONAL_EVIDENCE = "REQUEST_ADDITIONAL_EVIDENCE"

    # --------------------------------------------------------
    # MESSAGE TYPES
    # --------------------------------------------------------

    TYPE_REQUEST = "request"
    TYPE_RESPONSE = "response"
    TYPE_FEEDBACK = "feedback"

    def __init__(
        self,
        sender,
        receiver,
        message_type,
        request_type=None,
        task_type=None,
        payload=None,
        patient_id=None,
        correlation_id=None,
    ):

        self.message_id = str(uuid4())

        self.correlation_id = (
            correlation_id
            if correlation_id is not None
            else str(uuid4())
        )

        self.sender = sender
        self.receiver = receiver

        self.message_type = message_type
        self.request_type = request_type

        self.task_type = task_type
        self.patient_id = patient_id

        self.payload = payload or {}

        self.timestamp = datetime.utcnow().isoformat()

    # --------------------------------------------------------
    # SERIALIZATION
    # --------------------------------------------------------

    def to_dict(self):

        return {
            "message_id": self.message_id,
            "correlation_id": self.correlation_id,
            "sender": self.sender,
            "receiver": self.receiver,
            "message_type": self.message_type,
            "request_type": self.request_type,
            "task_type": self.task_type,
            "patient_id": self.patient_id,
            "payload": self.payload,
            "timestamp": self.timestamp,
        }

    # --------------------------------------------------------
    # REPRESENTATION
    # --------------------------------------------------------

    def __repr__(self):

        return (
            f"AgentMessage("
            f"{self.sender} -> {self.receiver}, "
            f"{self.request_type}, "
            f"{self.task_type})"
        )


class CommunicationProtocol:
    """
    Handles structured Coordinator <-> Agent communication.
    """

    def __init__(self):

        self.messages = []

    # --------------------------------------------------------
    # SEND REQUEST
    # --------------------------------------------------------

    def create_request(
        self,
        receiver,
        request_type,
        task_type=None,
        payload=None,
        patient_id=None,
    ):

        message = AgentMessage(
            sender="LiverCoordinator",
            receiver=receiver,
            message_type=AgentMessage.TYPE_REQUEST,
            request_type=request_type,
            task_type=task_type,
            payload=payload,
            patient_id=patient_id,
        )

        self.messages.append(message)

        return message

    # --------------------------------------------------------
    # CREATE RESPONSE
    # --------------------------------------------------------

    def create_response(
        self,
        receiver,
        request_message,
        payload=None,
        task_type=None,
        patient_id=None,
    ):

        message = AgentMessage(
            sender="LiverCoordinator",
            receiver=receiver,
            message_type=AgentMessage.TYPE_RESPONSE,
            request_type=request_message.request_type,
            task_type=task_type or request_message.task_type,
            payload=payload,
            patient_id=patient_id or request_message.patient_id,
            correlation_id=request_message.correlation_id,
        )

        self.messages.append(message)

        return message

    # --------------------------------------------------------
    # LOG AGENT RESPONSE
    # --------------------------------------------------------

    def record_agent_response(
        self,
        sender,
        request_message,
        payload=None,
        task_type=None,
        patient_id=None,
    ):

        message = AgentMessage(
            sender=sender,
            receiver="LiverCoordinator",
            message_type=AgentMessage.TYPE_RESPONSE,
            request_type=request_message.request_type,
            task_type=task_type or request_message.task_type,
            payload=payload,
            patient_id=patient_id or request_message.patient_id,
            correlation_id=request_message.correlation_id,
        )

        self.messages.append(message)

        return message

    # --------------------------------------------------------
    # GET COMMUNICATION TRACE
    # --------------------------------------------------------

    def get_trace(self):

        return [
            message.to_dict()
            for message in self.messages
        ]

    # --------------------------------------------------------
    # CLEAR
    # --------------------------------------------------------

    def clear(self):

        self.messages = []

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    def summary(self):

        return {
            "total_messages": len(self.messages),
            "requests": sum(
                1
                for m in self.messages
                if m.message_type == AgentMessage.TYPE_REQUEST
            ),
            "responses": sum(
                1
                for m in self.messages
                if m.message_type == AgentMessage.TYPE_RESPONSE
            ),
            "trace": self.get_trace(),
        }
