"""
agentkit quickstart — build a multi-agent app in 50 lines.

This example creates a simple weather + travel advisor using the framework.
No external APIs needed — just an ANTHROPIC_API_KEY in your environment.

Usage:
    pip install -e .
    export ANTHROPIC_API_KEY=sk-...
    python examples/quickstart.py
"""

import asyncio

import anthropic

from agentkit import (
    AgentContext,
    AgentResult,
    BaseAgent,
    Coordinator,
    InteractionLogger,
    ToolRegistry,
)

# 1. Create a registry and register tools
registry = ToolRegistry()


@registry.register("get_weather", "Get current weather for a city")
def get_weather(city: str) -> str:
    # Stub — replace with a real API call
    data = {
        "Austin": "92°F, sunny, humidity 45%",
        "Denver": "68°F, partly cloudy, humidity 30%",
        "Seattle": "58°F, rainy, humidity 80%",
    }
    return data.get(city, f"72°F, clear skies in {city}")


@registry.register("get_cost_of_living", "Get cost of living index for a city")
def get_cost_of_living(city: str) -> str:
    data = {
        "Austin": "Index 95 (near national average), median rent $1,800",
        "Denver": "Index 108 (above average), median rent $2,100",
        "Seattle": "Index 149 (well above average), median rent $2,400",
    }
    return data.get(city, f"Index 100 (average) for {city}")


# 2. Define a specialist agent
class CityAdvisor(BaseAgent):
    @property
    def name(self) -> str:
        return "CityAdvisor"

    @property
    def system_prompt(self) -> str:
        return (
            "You are a city living advisor. Use the available tools to gather "
            "weather and cost data, then give a concise recommendation. "
            "Cite specific numbers from the tools."
        )

    @property
    def tool_names(self) -> list[str]:
        return ["get_weather", "get_cost_of_living"]

    def build_user_message(self, context: AgentContext) -> str:
        return context.user_query

    async def parse_result(self, messages, context) -> AgentResult:
        # Extract final assistant text
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                content = msg["content"]
                if isinstance(content, list):
                    for block in content:
                        if hasattr(block, "text"):
                            return AgentResult(
                                agent_name=self.name,
                                content={"analysis": block.text},
                            )
                elif isinstance(content, str):
                    return AgentResult(
                        agent_name=self.name, content={"analysis": content}
                    )
        return AgentResult(agent_name=self.name, content={"analysis": ""})


# 3. Wire up and run
async def main():
    client = anthropic.AsyncAnthropic()
    logger = InteractionLogger()

    advisor = CityAdvisor(registry=registry, logger=logger, client=client)

    coordinator = Coordinator()
    coordinator.register_parallel(advisor)

    # No synthesizer needed for a single agent — just pass through
    coordinator.set_synthesizer(
        lambda ctx, par, seq: par[0].content["analysis"] if par else "No results."
    )

    context = AgentContext(
        session_id="quickstart",
        user_query="Should I move to Austin or Denver? Compare weather and cost of living.",
    )

    result = await coordinator.run(context)

    print("\n" + "=" * 60)
    print("ANSWER:")
    print("=" * 60)
    print(result["answer"])
    print(f"\nCost: ${result['total_cost']:.4f}")


if __name__ == "__main__":
    asyncio.run(main())
