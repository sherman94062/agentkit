"""agentkit — Reusable multi-agent framework for AI applications."""

from .base import BaseAgent
from .coordinator import Coordinator
from .eval import EvalCase, EvalHarness, EvalReport, EvalResult
from .exceptions import AgentError, GroundingError, ToolError
from .grounding import GroundingClaim, GroundingEvaluator, GroundingResult
from .logger import InteractionLogger
from .mcp import MCPServerBuilder
from .messages import AgentContext, AgentMessage, AgentResult, AgentRole
from .registry import ToolRegistry, ToolSpec

__version__ = "0.1.0"

__all__ = [
    "BaseAgent",
    "Coordinator",
    "ToolRegistry",
    "ToolSpec",
    "AgentContext",
    "AgentMessage",
    "AgentResult",
    "AgentRole",
    "InteractionLogger",
    "GroundingEvaluator",
    "GroundingClaim",
    "GroundingResult",
    "EvalHarness",
    "EvalCase",
    "EvalResult",
    "EvalReport",
    "MCPServerBuilder",
    "AgentError",
    "ToolError",
    "GroundingError",
]
