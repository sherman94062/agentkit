"""Coordinator — orchestrates multiple specialist agents."""

import asyncio
import logging
from typing import Callable

from .base import BaseAgent
from .messages import AgentContext, AgentResult

# Semaphore limits concurrent LLM calls to avoid rate limiting
_DEFAULT_CONCURRENCY = 2

logger = logging.getLogger("agentkit")


class Coordinator:
    """
    Orchestrates multiple specialist agents with parallel and sequential phases.

    Usage:
        coordinator = Coordinator()
        coordinator.register_parallel(market_agent, property_agent)
        coordinator.register_sequential(financial_agent)
        coordinator.set_synthesizer(synthesis_fn)
        result = await coordinator.run(context)
    """

    def __init__(self, max_concurrency: int = _DEFAULT_CONCURRENCY):
        self._parallel_agents: list[BaseAgent] = []
        self._sequential_agents: list[BaseAgent] = []
        self._synthesizer: Callable | None = None
        self._semaphore = asyncio.Semaphore(max_concurrency)

    def register_parallel(self, *agents: BaseAgent) -> None:
        self._parallel_agents.extend(agents)

    def register_sequential(self, *agents: BaseAgent) -> None:
        self._sequential_agents.extend(agents)

    def set_synthesizer(self, fn: Callable) -> None:
        """fn(context, parallel_results, sequential_results) -> str"""
        self._synthesizer = fn

    async def _run_agent_throttled(self, agent: BaseAgent, context: AgentContext) -> AgentResult:
        """Run an agent with concurrency limiting to avoid API rate limits."""
        async with self._semaphore:
            return await agent.run(context)

    async def run(self, context: AgentContext) -> dict:
        # Phase 1: parallel specialists (throttled by semaphore)
        parallel_results: list[AgentResult] = []
        if self._parallel_agents:
            raw = await asyncio.gather(
                *[self._run_agent_throttled(agent, context) for agent in self._parallel_agents],
                return_exceptions=True,
            )
            for i, result in enumerate(raw):
                if isinstance(result, Exception):
                    agent_name = self._parallel_agents[i].name
                    logger.error("Agent %s failed: %s", agent_name, result)
                else:
                    parallel_results.append(result)

        # Phase 2: sequential specialists (each can see prior results)
        sequential_results: list[AgentResult] = []
        for agent in self._sequential_agents:
            context.metadata["prior_results"] = [
                r.model_dump() for r in parallel_results + sequential_results
            ]
            try:
                result = await agent.run(context)
                sequential_results.append(result)
            except Exception as e:
                logger.error("Sequential agent %s failed: %s", agent.name, e)

        # Phase 3: synthesize
        all_results = parallel_results + sequential_results
        final_answer = ""
        if self._synthesizer:
            final_answer = await self._synthesizer(
                context, parallel_results, sequential_results
            )

        return {
            "answer": final_answer,
            "agent_results": all_results,
            "total_cost": sum(r.cost_usd for r in all_results),
            "total_tokens": sum(r.tokens_used for r in all_results),
        }
