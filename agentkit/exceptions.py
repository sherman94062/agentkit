"""agentkit exceptions."""


class AgentError(Exception):
    """Base exception for agent errors."""


class ToolError(AgentError):
    """Raised when a tool execution fails."""

    def __init__(self, tool_name: str, message: str):
        self.tool_name = tool_name
        super().__init__(f"Tool '{tool_name}' failed: {message}")


class GroundingError(AgentError):
    """Raised when grounding evaluation fails or score is below threshold."""

    def __init__(self, score: float, threshold: float):
        self.score = score
        self.threshold = threshold
        super().__init__(
            f"Grounding score {score:.2f} below threshold {threshold:.2f}"
        )
