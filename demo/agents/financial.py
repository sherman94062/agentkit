"""FinancialAgent — ROI, cap rate, cash flow, mortgage calculations."""

from agentkit import AgentContext, AgentResult, BaseAgent, ToolRegistry

from demo.agents.market import _extract_final_text, _extract_tool_results


class FinancialAgent(BaseAgent):

    @property
    def name(self) -> str:
        return "FinancialAgent"

    @property
    def system_prompt(self) -> str:
        return (
            "You are a real estate investment analyst. Show all calculations clearly. "
            "State your assumptions. Flag when key inputs (rent estimate, tax, insurance) "
            "are estimated vs. known. Use the financial tools to compute mortgage payments, "
            "cash flow, and cap rate. Present a clear investment summary."
        )

    @property
    def tool_names(self) -> list[str]:
        return [
            "get_current_mortgage_rates",
            "calculate_mortgage",
            "estimate_rental_income",
            "calculate_cash_flow",
            "calculate_cap_rate",
        ]

    def build_user_message(self, context: AgentContext) -> str:
        address = context.metadata.get("address", "")
        prior = context.metadata.get("prior_results", [])
        parts = [context.user_query]
        if address:
            parts.append(f"\nProperty address: {address}")
        if prior:
            parts.append("\n--- Data from other agents ---")
            for r in prior:
                name = r.get("agent_name", "unknown")
                content = r.get("content", {})
                analysis = content.get("analysis", "")
                if analysis:
                    parts.append(f"\n[{name}]:\n{analysis[:500]}")
        return "\n".join(parts)

    async def parse_result(self, messages: list[dict], context: AgentContext) -> AgentResult:
        text = _extract_final_text(messages)
        sources = _extract_tool_results(messages)
        usage = self.interaction_logger.total_tokens
        return AgentResult(
            agent_name=self.name,
            content={"analysis": text},
            sources=sources,
            confidence=0.9 if sources else 0.5,
            tokens_used=usage[0] + usage[1],
            cost_usd=self.interaction_logger.total_cost,
        )


# ---------------------------------------------------------------------------
# Pure-Python calculation tools (no external API)
# ---------------------------------------------------------------------------


def _calculate_mortgage(
    price: float, down_pct: float, rate: float, term_years: int = 30
) -> dict:
    """Calculate monthly mortgage payment and amortization summary."""
    down = price * (down_pct / 100)
    loan = price - down
    monthly_rate = (rate / 100) / 12
    n_payments = term_years * 12

    if monthly_rate == 0:
        monthly_payment = loan / n_payments
    else:
        monthly_payment = loan * (monthly_rate * (1 + monthly_rate) ** n_payments) / (
            (1 + monthly_rate) ** n_payments - 1
        )

    total_paid = monthly_payment * n_payments
    total_interest = total_paid - loan

    return {
        "purchase_price": price,
        "down_payment": round(down, 2),
        "down_payment_pct": down_pct,
        "loan_amount": round(loan, 2),
        "interest_rate": rate,
        "term_years": term_years,
        "monthly_payment": round(monthly_payment, 2),
        "total_paid": round(total_paid, 2),
        "total_interest": round(total_interest, 2),
    }


def _calculate_cash_flow(
    purchase_price: float,
    down_pct: float,
    rate: float,
    monthly_rent: float,
    tax_annual: float,
    insurance_annual: float,
    vacancy_rate: float = 5.0,
    management_pct: float = 10.0,
) -> dict:
    """Calculate monthly and annual cash flow for a rental property."""
    mortgage = _calculate_mortgage(purchase_price, down_pct, rate)
    monthly_mortgage = mortgage["monthly_payment"]

    effective_rent = monthly_rent * (1 - vacancy_rate / 100)
    management = monthly_rent * (management_pct / 100)
    monthly_tax = tax_annual / 12
    monthly_insurance = insurance_annual / 12

    total_expenses = monthly_mortgage + management + monthly_tax + monthly_insurance
    monthly_cash_flow = effective_rent - total_expenses
    annual_cash_flow = monthly_cash_flow * 12
    cash_on_cash = (annual_cash_flow / mortgage["down_payment"] * 100) if mortgage["down_payment"] > 0 else 0

    return {
        "monthly_rent": monthly_rent,
        "effective_rent": round(effective_rent, 2),
        "vacancy_rate_pct": vacancy_rate,
        "monthly_mortgage": monthly_mortgage,
        "monthly_management": round(management, 2),
        "monthly_tax": round(monthly_tax, 2),
        "monthly_insurance": round(monthly_insurance, 2),
        "total_monthly_expenses": round(total_expenses, 2),
        "monthly_cash_flow": round(monthly_cash_flow, 2),
        "annual_cash_flow": round(annual_cash_flow, 2),
        "cash_on_cash_return_pct": round(cash_on_cash, 2),
        "down_payment": mortgage["down_payment"],
    }


def _calculate_cap_rate(noi: float, purchase_price: float) -> dict:
    """Calculate capitalization rate."""
    if purchase_price <= 0:
        return {"error": "Purchase price must be positive"}
    cap_rate = (noi / purchase_price) * 100
    return {
        "noi": round(noi, 2),
        "purchase_price": purchase_price,
        "cap_rate_pct": round(cap_rate, 2),
    }


def register_tools(registry: ToolRegistry) -> None:
    """Register FinancialAgent tools."""
    from demo.ingest import fred, rentcast

    registry.add(
        "get_current_mortgage_rates",
        fred.get_current_mortgage_rates,
        "Get the current 30-year fixed mortgage rate from FRED (Federal Reserve).",
        tags=["external", "read"],
        input_schema={"type": "object", "properties": {}, "required": []},
    )

    registry.add(
        "calculate_mortgage",
        _calculate_mortgage,
        "Calculate monthly mortgage payment given price, down payment %, rate, and term.",
        tags=["compute"],
        input_schema={
            "type": "object",
            "properties": {
                "price": {"type": "number", "description": "Purchase price in dollars"},
                "down_pct": {"type": "number", "description": "Down payment percentage (e.g. 20 for 20%)"},
                "rate": {"type": "number", "description": "Annual interest rate (e.g. 6.5 for 6.5%)"},
                "term_years": {"type": "integer", "description": "Loan term in years", "default": 30},
            },
            "required": ["price", "down_pct", "rate"],
        },
    )

    registry.add(
        "estimate_rental_income",
        rentcast.estimate_rent,
        "Estimate monthly rental income for a property based on address and beds/baths.",
        tags=["external", "read"],
        input_schema={
            "type": "object",
            "properties": {
                "address": {"type": "string", "description": "Property address"},
                "bedrooms": {"type": "integer", "description": "Number of bedrooms", "default": 3},
                "bathrooms": {"type": "number", "description": "Number of bathrooms", "default": 2.0},
            },
            "required": ["address"],
        },
    )

    registry.add(
        "calculate_cash_flow",
        _calculate_cash_flow,
        "Calculate monthly cash flow for a rental: income minus mortgage, tax, insurance, management, vacancy.",
        tags=["compute"],
        input_schema={
            "type": "object",
            "properties": {
                "purchase_price": {"type": "number", "description": "Purchase price"},
                "down_pct": {"type": "number", "description": "Down payment %"},
                "rate": {"type": "number", "description": "Annual interest rate %"},
                "monthly_rent": {"type": "number", "description": "Expected monthly rent"},
                "tax_annual": {"type": "number", "description": "Annual property tax"},
                "insurance_annual": {"type": "number", "description": "Annual insurance cost"},
                "vacancy_rate": {"type": "number", "description": "Vacancy rate %", "default": 5.0},
                "management_pct": {"type": "number", "description": "Property management fee %", "default": 10.0},
            },
            "required": ["purchase_price", "down_pct", "rate", "monthly_rent", "tax_annual", "insurance_annual"],
        },
    )

    registry.add(
        "calculate_cap_rate",
        _calculate_cap_rate,
        "Calculate cap rate from net operating income (NOI) and purchase price.",
        tags=["compute"],
        input_schema={
            "type": "object",
            "properties": {
                "noi": {"type": "number", "description": "Annual net operating income"},
                "purchase_price": {"type": "number", "description": "Purchase price"},
            },
            "required": ["noi", "purchase_price"],
        },
    )
