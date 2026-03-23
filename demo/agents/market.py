"""MarketAgent — listings, comps, price trends, days on market."""

import json

from agentkit import AgentContext, AgentResult, BaseAgent, InteractionLogger, ToolRegistry

import anthropic


class MarketAgent(BaseAgent):

    @property
    def name(self) -> str:
        return "MarketAgent"

    @property
    def system_prompt(self) -> str:
        return (
            "You are a real estate market analyst. Use listing and comparable sale "
            "data to characterize market conditions. Always cite specific data points "
            "(price per sq ft, days on market, list-to-sale ratio). "
            "Be concise and data-driven. If data is unavailable, say so explicitly."
        )

    @property
    def tool_names(self) -> list[str]:
        return ["search_listings", "get_comps", "get_market_stats"]

    def build_user_message(self, context: AgentContext) -> str:
        address = context.metadata.get("address", "")
        parts = [context.user_query]
        if address:
            parts.append(f"\nProperty address: {address}")
        return "\n".join(parts)

    async def parse_result(self, messages: list[dict], context: AgentContext) -> AgentResult:
        # Extract the final assistant text
        text = _extract_final_text(messages)
        sources = _extract_tool_results(messages)

        usage = self.interaction_logger.total_tokens
        cost = self.interaction_logger.total_cost

        return AgentResult(
            agent_name=self.name,
            content={"analysis": text},
            sources=sources,
            confidence=0.8 if sources else 0.4,
            tokens_used=usage[0] + usage[1],
            cost_usd=cost,
        )


def register_tools(registry: ToolRegistry) -> None:
    """Register MarketAgent tools into the shared registry."""
    from demo.ingest import rentcast

    registry.add(
        "search_listings",
        rentcast.search_listings,
        "Search active sale listings near an address. Returns price, sqft, beds, baths, days on market.",
        tags=["external", "read"],
        input_schema={
            "type": "object",
            "properties": {
                "address": {"type": "string", "description": "Property address or area to search near"},
                "radius_miles": {"type": "number", "description": "Search radius in miles", "default": 5.0},
                "max_results": {"type": "integer", "description": "Maximum listings to return", "default": 10},
            },
            "required": ["address"],
        },
    )

    registry.add(
        "get_comps",
        rentcast.get_comps,
        "Get comparable recent sales near an address. Returns sale price, date, sqft, price/sqft.",
        tags=["external", "read"],
        input_schema={
            "type": "object",
            "properties": {
                "address": {"type": "string", "description": "Subject property address"},
                "radius_miles": {"type": "number", "description": "Search radius in miles", "default": 5.0},
                "sold_in_days": {"type": "integer", "description": "Only include sales within this many days", "default": 180},
            },
            "required": ["address"],
        },
    )

    registry.add(
        "get_market_stats",
        rentcast.get_market_stats,
        "Get market statistics for a ZIP code: median price, inventory, days on market, price trends.",
        tags=["external", "read"],
        input_schema={
            "type": "object",
            "properties": {
                "zipcode": {"type": "string", "description": "5-digit ZIP code"},
            },
            "required": ["zipcode"],
        },
    )


def _extract_final_text(messages: list[dict]) -> str:
    """Get the last assistant text from the message history."""
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            content = msg.get("content", "")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                for block in content:
                    if hasattr(block, "text"):
                        return block.text
    return ""


def _extract_tool_results(messages: list[dict]) -> list[dict]:
    """Extract tool results as sources."""
    sources = []
    for msg in messages:
        content = msg.get("content", [])
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                sources.append({"text": str(block.get("content", ""))[:500]})
    return sources
