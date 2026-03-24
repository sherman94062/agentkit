"""MCP server for the real estate demo — exposes the coordinator to Claude Desktop.

Usage:
    python -m demo.mcp_server

Add to ~/.claude/claude_desktop_config.json:
{
    "mcpServers": {
        "real-estate": {
            "command": "python",
            "args": ["-m", "demo.mcp_server"],
            "cwd": "/path/to/agentkit",
            "env": {
                "ANTHROPIC_API_KEY": "...",
                "RENTCAST_API_KEY": "...",
                "ATTOM_API_KEY": "...",
                "WALKSCORE_API_KEY": "..."
            }
        }
    }
}
"""

import uuid

from fastmcp import FastMCP

import anthropic

from agentkit import AgentContext, GroundingEvaluator
from demo.agents.coordinator import build_coordinator
from demo.db import client as db

# Initialize
db.init_db()
client = anthropic.AsyncAnthropic()
coordinator, registry, logger = build_coordinator(client)

mcp = FastMCP("real-estate-intelligence")


@mcp.tool()
async def query_property(
    question: str,
    property_address: str = "",
    session_id: str = "",
) -> str:
    """Ask a question about a property or real estate market.

    Examples:
        - "Is 1404 Lions Lair, Leander TX fairly priced?"
        - "What would my cash flow be at 20% down?"
        - "Compare school ratings near 78641"
    """
    ctx = AgentContext(
        session_id=session_id or uuid.uuid4().hex,
        user_query=question,
        metadata={"address": property_address} if property_address else {},
    )
    result = await coordinator.run(ctx)
    return result.get("answer", "No answer generated.")


@mcp.tool()
async def calculate_mortgage(
    price: float,
    down_payment_pct: float = 20.0,
    interest_rate: float = 6.85,
    term_years: int = 30,
) -> str:
    """Calculate monthly mortgage payment."""
    from demo.agents.financial import _calculate_mortgage

    result = _calculate_mortgage(price, down_payment_pct, interest_rate, term_years)
    return (
        f"Purchase: ${result['purchase_price']:,.0f}\n"
        f"Down: ${result['down_payment']:,.0f} ({result['down_payment_pct']}%)\n"
        f"Loan: ${result['loan_amount']:,.0f}\n"
        f"Rate: {result['interest_rate']}% / {result['term_years']}yr\n"
        f"Monthly payment: ${result['monthly_payment']:,.2f}\n"
        f"Total interest: ${result['total_interest']:,.0f}"
    )


@mcp.tool()
async def calculate_investment(
    purchase_price: float,
    monthly_rent: float,
    down_payment_pct: float = 20.0,
    interest_rate: float = 6.85,
    annual_tax: float = 6000.0,
    annual_insurance: float = 2400.0,
    vacancy_pct: float = 5.0,
    management_pct: float = 10.0,
) -> str:
    """Analyze a rental property investment: cash flow, cap rate, cash-on-cash return."""
    from demo.agents.financial import _calculate_cash_flow, _calculate_cap_rate

    cf = _calculate_cash_flow(
        purchase_price, down_payment_pct, interest_rate,
        monthly_rent, annual_tax, annual_insurance,
        vacancy_pct, management_pct,
    )
    noi = (monthly_rent * 12 * (1 - vacancy_pct / 100)) - annual_tax - annual_insurance - (monthly_rent * 12 * management_pct / 100)
    cr = _calculate_cap_rate(noi, purchase_price)

    return (
        f"Monthly rent: ${cf['monthly_rent']:,.0f}\n"
        f"Effective rent: ${cf['effective_rent']:,.0f}/mo (after {cf['vacancy_rate_pct']}% vacancy)\n"
        f"Total expenses: ${cf['total_monthly_expenses']:,.2f}/mo\n"
        f"Monthly cash flow: ${cf['monthly_cash_flow']:,.2f}\n"
        f"Annual cash flow: ${cf['annual_cash_flow']:,.2f}\n"
        f"Cash-on-cash return: {cf['cash_on_cash_return_pct']:.2f}%\n"
        f"Cap rate: {cr['cap_rate_pct']:.2f}%"
    )


@mcp.tool()
async def list_saved_properties() -> str:
    """List all saved properties in the database."""
    import json
    props = db.get_properties()
    if not props:
        return "No saved properties."
    lines = []
    for p in props:
        meta = json.loads(p.get("metadata_json", "{}"))
        assessed = meta.get("assessed_value_2025", "N/A")
        lines.append(
            f"[{p['id']}] {p['address']} — {p['property_type']}, "
            f"assessed: ${assessed:,}" if isinstance(assessed, int) else
            f"[{p['id']}] {p['address']} — {p['property_type']}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    mcp.run()
