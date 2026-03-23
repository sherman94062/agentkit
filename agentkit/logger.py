"""Interaction logger — audit trail for all LLM calls."""

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("agentkit")


@dataclass
class LogEntry:
    agent_name: str
    timestamp: float
    messages_sent: list[dict]
    response_summary: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    duration_s: float


class InteractionLogger:
    """Records every LLM call for audit and debugging."""

    # Approximate pricing per 1M tokens (Claude Sonnet 4)
    INPUT_COST_PER_M = 3.00
    OUTPUT_COST_PER_M = 15.00

    def __init__(self):
        self._entries: list[LogEntry] = []
        self._call_start: float | None = None

    def start_call(self) -> None:
        self._call_start = time.monotonic()

    def log_llm_call(
        self,
        agent_name: str,
        messages: list[dict],
        response: Any,
    ) -> LogEntry:
        duration = 0.0
        if self._call_start is not None:
            duration = time.monotonic() - self._call_start
            self._call_start = None

        input_tokens = getattr(response.usage, "input_tokens", 0)
        output_tokens = getattr(response.usage, "output_tokens", 0)
        cost = (
            input_tokens * self.INPUT_COST_PER_M / 1_000_000
            + output_tokens * self.OUTPUT_COST_PER_M / 1_000_000
        )

        # Build a short summary of the response
        summary = ""
        if hasattr(response, "content"):
            for block in response.content:
                if hasattr(block, "text"):
                    summary = block.text[:200]
                    break
                if hasattr(block, "type") and block.type == "tool_use":
                    summary = f"tool_use: {block.name}"
                    break

        entry = LogEntry(
            agent_name=agent_name,
            timestamp=time.time(),
            messages_sent=[{"role": m.get("role", "?")} for m in messages],
            response_summary=summary,
            model=getattr(response, "model", "unknown"),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            duration_s=duration,
        )
        self._entries.append(entry)
        logger.debug(
            "LLM call: agent=%s tokens=%d+%d cost=$%.4f",
            agent_name,
            input_tokens,
            output_tokens,
            cost,
        )
        return entry

    @property
    def entries(self) -> list[LogEntry]:
        return list(self._entries)

    @property
    def total_cost(self) -> float:
        return sum(e.cost_usd for e in self._entries)

    @property
    def total_tokens(self) -> tuple[int, int]:
        return (
            sum(e.input_tokens for e in self._entries),
            sum(e.output_tokens for e in self._entries),
        )

    def to_dicts(self) -> list[dict]:
        return [
            {
                "agent": e.agent_name,
                "timestamp": e.timestamp,
                "model": e.model,
                "input_tokens": e.input_tokens,
                "output_tokens": e.output_tokens,
                "cost_usd": e.cost_usd,
                "duration_s": e.duration_s,
                "summary": e.response_summary,
            }
            for e in self._entries
        ]
