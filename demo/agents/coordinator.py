"""RealEstateCoordinator — orchestrates all specialist agents."""

import json

import anthropic

from agentkit import (
    AgentContext,
    AgentResult,
    Coordinator,
    GroundingEvaluator,
    InteractionLogger,
    ToolRegistry,
)

from demo.agents.market import MarketAgent, register_tools as register_market
from demo.agents.property import PropertyAgent, register_tools as register_property
from demo.agents.neighborhood import NeighborhoodAgent, register_tools as register_neighborhood
from demo.agents.financial import FinancialAgent, register_tools as register_financial
from demo.agents.document import DocumentAgent, register_tools as register_document


SYNTHESIS_PROMPT = """You are synthesizing results from multiple real estate specialist agents
into a single coherent answer for the user.

USER QUESTION: {query}

AGENT RESULTS:
{agent_results}

Instructions:
1. Combine all agent findings into a coherent 3-5 paragraph answer.
2. Include a "Bottom Line" section with a clear recommendation or summary.
3. Cite specific data points from each agent (prices, scores, rates, etc.).
4. If agents provided conflicting information, note the discrepancy.
5. If critical data was unavailable, mention what's missing.

Respond with the synthesized answer only — no preamble."""


async def _synthesize(
    context: AgentContext,
    parallel_results: list[AgentResult],
    sequential_results: list[AgentResult],
    client: anthropic.AsyncAnthropic,
) -> str:
    """Combine all agent results into a single answer using Claude."""
    all_results = parallel_results + sequential_results
    if not all_results:
        return "No agent results were available to synthesize an answer."

    results_text = "\n\n".join(
        f"[{r.agent_name}]:\n{r.content.get('analysis', json.dumps(r.content))}"
        for r in all_results
    )

    prompt = SYNTHESIS_PROMPT.format(query=context.user_query, agent_results=results_text)

    response = await client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def build_coordinator(
    client: anthropic.AsyncAnthropic | None = None,
) -> tuple[Coordinator, ToolRegistry, InteractionLogger]:
    """
    Build a fully-wired RealEstateCoordinator.

    Returns (coordinator, registry, logger) so the caller can inspect them.
    """
    if client is None:
        client = anthropic.AsyncAnthropic()

    registry = ToolRegistry()
    logger = InteractionLogger()

    # Register all tools
    register_market(registry)
    register_property(registry)
    register_neighborhood(registry)
    register_financial(registry)
    register_document(registry)

    # Create specialist agents
    market = MarketAgent(registry, logger, client)
    prop = PropertyAgent(registry, logger, client)
    neighborhood = NeighborhoodAgent(registry, logger, client)
    document = DocumentAgent(registry, logger, client)
    financial = FinancialAgent(registry, logger, client)

    # Wire up coordinator
    coord = Coordinator()
    coord.register_parallel(market, prop, neighborhood, document)
    coord.register_sequential(financial)  # runs after parallel — sees their results

    # Synthesis function captures the client
    async def synthesize(ctx, par, seq):
        return await _synthesize(ctx, par, seq, client)

    coord.set_synthesizer(synthesize)

    return coord, registry, logger


async def run_with_grounding(
    coordinator: Coordinator,
    context: AgentContext,
    client: anthropic.AsyncAnthropic,
) -> dict:
    """Run the coordinator and add grounding evaluation."""
    result = await coordinator.run(context)

    # Grounding check
    all_sources = []
    for agent_result in result.get("agent_results", []):
        all_sources.extend(agent_result.sources)

    grounding_score = 0.0
    if result.get("answer") and all_sources:
        evaluator = GroundingEvaluator(client)
        grounding = await evaluator.evaluate(result["answer"], all_sources)
        grounding_score = grounding.score
        result["grounding"] = grounding.model_dump()

    result["grounding_score"] = grounding_score
    return result
