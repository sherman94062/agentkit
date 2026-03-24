"""BaseAgent — async tool-use loop that every specialist inherits from."""

import asyncio
import logging
from abc import ABC, abstractmethod

import anthropic

from .exceptions import AgentError
from .logger import InteractionLogger
from .messages import AgentContext, AgentResult
from .registry import ToolRegistry

logger = logging.getLogger("agentkit")


class BaseAgent(ABC):
    """
    Abstract base for all specialist agents.

    Subclass and implement:
      - name (property)
      - system_prompt (property)
      - tool_names (property)
      - build_user_message(context) -> str
      - parse_result(messages, context) -> AgentResult
    """

    MODEL = "claude-sonnet-4-20250514"
    MAX_TOKENS = 4096
    MAX_ROUNDS = 5
    MAX_RETRIES = 3

    def __init__(
        self,
        registry: ToolRegistry,
        logger: InteractionLogger,
        client: anthropic.AsyncAnthropic,
    ):
        self.registry = registry
        self.interaction_logger = logger
        self.client = client

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def system_prompt(self) -> str: ...

    @property
    @abstractmethod
    def tool_names(self) -> list[str]: ...

    @abstractmethod
    def build_user_message(self, context: AgentContext) -> str: ...

    @abstractmethod
    async def parse_result(
        self, messages: list[dict], context: AgentContext
    ) -> AgentResult: ...

    async def run(self, context: AgentContext) -> AgentResult:
        """Main entry point. Runs the async tool-use loop."""
        messages = [{"role": "user", "content": self.build_user_message(context)}]
        tools = self.registry.to_anthropic_tools(self.tool_names)

        for round_num in range(self.MAX_ROUNDS):
            self.interaction_logger.start_call()
            response = await self._call_api(messages, tools)
            self.interaction_logger.log_llm_call(self.name, messages, response)

            if response.stop_reason == "end_turn":
                messages.append({"role": "assistant", "content": response.content})
                break

            if response.stop_reason == "tool_use":
                messages.append({"role": "assistant", "content": response.content})
                tool_results = await self._execute_tools(response.content)
                messages.append({"role": "user", "content": tool_results})
                continue

            # Unexpected stop reason
            logger.warning(
                "Agent %s got unexpected stop_reason=%s at round %d",
                self.name,
                response.stop_reason,
                round_num,
            )
            messages.append({"role": "assistant", "content": response.content})
            break
        else:
            logger.warning("Agent %s hit MAX_ROUNDS=%d", self.name, self.MAX_ROUNDS)

        return await self.parse_result(messages, context)

    async def _call_api(self, messages, tools, retries=0):
        try:
            return await self.client.messages.create(
                model=self.MODEL,
                max_tokens=self.MAX_TOKENS,
                system=self.system_prompt,
                tools=tools,
                messages=messages,
            )
        except anthropic.RateLimitError:
            if retries >= self.MAX_RETRIES:
                raise
            wait = 2**retries
            logger.info("Rate limited, retrying in %ds (attempt %d)", wait, retries + 1)
            await asyncio.sleep(wait)
            return await self._call_api(messages, tools, retries + 1)
        except anthropic.APIError as e:
            raise AgentError(f"API error in agent '{self.name}': {e}") from e

    async def _execute_tools(self, content_blocks) -> list[dict]:
        """Execute all tool_use blocks in parallel, return tool_result blocks."""
        calls = [b for b in content_blocks if b.type == "tool_use"]
        results = await asyncio.gather(
            *[self._execute_one_tool(c) for c in calls],
            return_exceptions=True,
        )
        tool_results = []
        for call, result in zip(calls, results):
            if isinstance(result, Exception):
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": call.id,
                        "content": f"Error: {result}",
                        "is_error": True,
                    }
                )
            else:
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": call.id,
                        "content": str(result),
                    }
                )
        return tool_results

    async def _execute_one_tool(self, tool_use_block):
        spec = self.registry.get(tool_use_block.name)
        logger.debug("Executing tool %s with input %s", spec.name, tool_use_block.input)
        if asyncio.iscoroutinefunction(spec.fn):
            return await spec.fn(**tool_use_block.input)
        else:
            return spec.fn(**tool_use_block.input)
