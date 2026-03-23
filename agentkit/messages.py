"""Typed inter-agent message models."""

from enum import Enum
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class AgentRole(str, Enum):
    COORDINATOR = "coordinator"
    SPECIALIST = "specialist"
    EVALUATOR = "evaluator"


class AgentMessage(BaseModel):
    """A message sent between agents."""

    sender: str
    recipient: str
    content: dict[str, Any]
    message_id: str = Field(default_factory=lambda: uuid4().hex)
    parent_id: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentContext(BaseModel):
    """Shared context passed into every agent call."""

    session_id: str
    user_query: str
    history: list[AgentMessage] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentResult(BaseModel):
    """What a specialist agent returns."""

    agent_name: str
    content: dict[str, Any]
    sources: list[dict] = Field(default_factory=list)
    confidence: Optional[float] = None
    coverage_notes: str = ""
    tokens_used: int = 0
    cost_usd: float = 0.0
