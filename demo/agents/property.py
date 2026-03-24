"""PropertyAgent — parcel details, tax history, ownership, zoning."""

from agentkit import AgentContext, AgentResult, BaseAgent, ToolRegistry

from demo.agents.market import _extract_final_text, _extract_tool_results


class PropertyAgent(BaseAgent):
    MAX_TOKENS = 2048

    @property
    def name(self) -> str:
        return "PropertyAgent"

    @property
    def system_prompt(self) -> str:
        return (
            "You are a property records analyst. Report factual data about parcel "
            "characteristics, tax assessments, and ownership history. Note any unusual "
            "patterns in tax history or ownership transfers. Always cite the data source."
        )

    @property
    def tool_names(self) -> list[str]:
        return ["get_parcel", "get_tax_history", "get_sale_history", "get_avm"]

    def build_user_message(self, context: AgentContext) -> str:
        address = context.metadata.get("address", "")
        parts = [context.user_query]
        if address:
            parts.append(f"\nProperty address: {address}")
        return "\n".join(parts)

    async def parse_result(self, messages: list[dict], context: AgentContext) -> AgentResult:
        text = _extract_final_text(messages)
        sources = _extract_tool_results(messages)
        usage = self.interaction_logger.total_tokens
        return AgentResult(
            agent_name=self.name,
            content={"analysis": text},
            sources=sources,
            confidence=0.85 if sources else 0.3,
            tokens_used=usage[0] + usage[1],
            cost_usd=self.interaction_logger.total_cost,
        )


def register_tools(registry: ToolRegistry) -> None:
    """Register PropertyAgent tools into the shared registry."""
    from demo.ingest import attom

    registry.add(
        "get_parcel",
        attom.get_parcel,
        "Get basic property profile: lot size, bedrooms, bathrooms, year built, zoning, building sqft.",
        tags=["external", "read"],
        input_schema={
            "type": "object",
            "properties": {
                "address": {"type": "string", "description": "Full property address"},
            },
            "required": ["address"],
        },
    )

    registry.add(
        "get_tax_history",
        attom.get_tax_history,
        "Get tax assessment history: assessed value, tax amount, year-over-year changes.",
        tags=["external", "read"],
        input_schema={
            "type": "object",
            "properties": {
                "attom_id": {"type": "string", "description": "ATTOM property ID"},
            },
            "required": ["attom_id"],
        },
    )

    registry.add(
        "get_sale_history",
        attom.get_sale_history,
        "Get sale/transfer history: prior sale prices, dates, buyer/seller names.",
        tags=["external", "read"],
        input_schema={
            "type": "object",
            "properties": {
                "attom_id": {"type": "string", "description": "ATTOM property ID"},
            },
            "required": ["attom_id"],
        },
    )

    registry.add(
        "get_avm",
        attom.get_avm,
        "Get automated valuation model (AVM) estimate for a property: estimated value, confidence range.",
        tags=["external", "read"],
        input_schema={
            "type": "object",
            "properties": {
                "address": {"type": "string", "description": "Full property address"},
            },
            "required": ["address"],
        },
    )
