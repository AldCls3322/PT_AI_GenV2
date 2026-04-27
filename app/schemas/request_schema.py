"""
Description of API requests, use Pydantic models
JSON results only valid.
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


class MessagePayload(BaseModel):
    text: str = Field(..., min_length=1, description="User's message text") #? extract all parameters


class Metadata(BaseModel):
    channel: str = Field(
        default="api",
        description="Origin channel: 'app', 'web', 'whatsapp', 'ivr', etc."
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="ISO-8601 timestamp of the request"
    )


class ChatRequest(BaseModel):
    """
    {
        "conversation_id": string,
        "user_id": string,
        "message": { "text": string },
        "metadata": { "channel": <string>list, "timestamp": "2024-06-01T10:00:00Z" }
    }
    """
    conversation_id: str = Field(..., description="Unique conversation / session ID")
    user_id: str = Field(..., description="Authenticated bank client ID")
    message: MessagePayload
    metadata: Optional[Metadata] = Field(default_factory=Metadata)


class ProcessInfo(BaseModel):
    """Serialized row from functional_process table, attached to responses."""
    process_id: int
    process_name: str
    team_area_responsible: str
    average_time_till_solved: str
    atention_channel: str
    priority_level: str


class ChatResponse(BaseModel):
    """Response of chat request call."""
    conversation_id: str
    user_id: str
    reply: str = Field(..., description="Agent's natural-language response")
    detected_process: Optional[str] = Field(
        None, description="Process label detected by the agent (A-E)"
    )
    process_info: Optional[ProcessInfo] = Field(
        None, description="DB metadata about the detected process"
    )
    sources: list[str] = Field(
        default_factory=list,
        description="Document sources used by the RAG retriever"
    )
    timestamp: datetime = Field(default_factory=datetime.utcnow)
