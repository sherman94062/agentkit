"""NeighborhoodAgent — walk/transit/bike scores, demographics, flood zone."""

from agentkit import AgentContext, AgentResult, BaseAgent, ToolRegistry

from demo.agents.market import _extract_final_text, _extract_tool_results


class NeighborhoodAgent(BaseAgent):
    MODEL = "claude-haiku-4-5-20251001"
    MAX_TOKENS = 2048

    @property
    def name(self) -> str:
        return "NeighborhoodAgent"

    @property
    def system_prompt(self) -> str:
        return (
            "You are a neighborhood research analyst. Provide objective scores and data "
            "about livability, school quality, commute options, and environmental risks. "
            "Present data clearly with specific numbers. Note when data is estimated vs. measured."
        )

    @property
    def tool_names(self) -> list[str]:
        return ["get_walk_score", "get_demographics"]

    def build_user_message(self, context: AgentContext) -> str:
        address = context.metadata.get("address", "")
        lat = context.metadata.get("lat", "")
        lon = context.metadata.get("lon", "")
        parts = [context.user_query]
        if address:
            parts.append(f"\nProperty address: {address}")
        if lat and lon:
            parts.append(f"Coordinates: {lat}, {lon}")
        return "\n".join(parts)

    async def parse_result(self, messages: list[dict], context: AgentContext) -> AgentResult:
        text = _extract_final_text(messages)
        sources = _extract_tool_results(messages)
        usage = self.interaction_logger.total_tokens
        return AgentResult(
            agent_name=self.name,
            content={"analysis": text},
            sources=sources,
            confidence=0.8 if sources else 0.3,
            tokens_used=usage[0] + usage[1],
            cost_usd=self.interaction_logger.total_cost,
        )


def register_tools(registry: ToolRegistry) -> None:
    """Register NeighborhoodAgent tools."""
    from demo.ingest import walkscore, census

    registry.add(
        "get_walk_score",
        walkscore.get_walk_score,
        "Get Walk Score, Transit Score, and Bike Score for a location (0-100 each).",
        tags=["external", "read"],
        input_schema={
            "type": "object",
            "properties": {
                "address": {"type": "string", "description": "Property address"},
                "lat": {"type": "number", "description": "Latitude"},
                "lon": {"type": "number", "description": "Longitude"},
            },
            "required": ["address", "lat", "lon"],
        },
    )

    registry.add(
        "get_demographics",
        census.get_demographics,
        "Get Census demographics for a ZIP: population, median income, median age, home values, rent.",
        tags=["external", "read"],
        input_schema={
            "type": "object",
            "properties": {
                "zipcode": {"type": "string", "description": "5-digit ZIP code"},
            },
            "required": ["zipcode"],
        },
    )
