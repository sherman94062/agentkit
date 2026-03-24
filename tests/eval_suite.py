"""
Eval suite — 10 test cases for the real estate multi-agent pipeline.

Usage:
    python tests/eval_suite.py

Requires ANTHROPIC_API_KEY and ideally data API keys in .env.
Target: 8/10 pass, average grounding >= 75%.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import anthropic

from agentkit import EvalCase, EvalHarness
from demo.agents.coordinator import build_coordinator
from demo.db import client as db

EVAL_CASES = [
    EvalCase(
        name="basic_valuation",
        query="Is 1404 Lions Lair, Leander TX 78641 fairly priced at $1.1M based on comps and market data?",
        expected_keywords=["comparable", "price", "market"],
        expected_sources=["MarketAgent", "PropertyAgent"],
        metadata={"address": "1404 Lions Lair, Leander, TX 78641"},
    ),
    EvalCase(
        name="investment_cashflow",
        query="What would my monthly cash flow be if I built a $450k home on a vacant lot and rented it at $3200/mo with 20% down?",
        expected_keywords=["cash flow", "mortgage", "rent"],
        expected_sources=["FinancialAgent"],
    ),
    EvalCase(
        name="neighborhood_demographics",
        query="What are the demographics and livability scores near ZIP 78641?",
        expected_keywords=["population", "income", "score"],
        expected_sources=["NeighborhoodAgent"],
        metadata={"address": "Leander, TX 78641"},
    ),
    EvalCase(
        name="land_valuation",
        query="What is the per-acre value of 5-acre lots in Blanco County TX near Round Mountain?",
        expected_keywords=["acre", "Blanco", "value"],
        expected_sources=["MarketAgent", "PropertyAgent"],
        metadata={"address": "Round Mountain, TX 78636"},
    ),
    EvalCase(
        name="rate_impact",
        query="How does today's 30-year mortgage rate affect affordability on a $400k home compared to 12 months ago?",
        expected_keywords=["rate", "payment", "afford"],
        expected_sources=["FinancialAgent"],
    ),
    EvalCase(
        name="tax_history",
        query="What is the tax assessment history for 1404 Lions Lair, Leander TX?",
        expected_keywords=["tax", "assessed", "value"],
        expected_sources=["PropertyAgent"],
        metadata={"address": "1404 Lions Lair, Leander, TX 78641"},
    ),
    EvalCase(
        name="cap_rate_analysis",
        query="Calculate the cap rate for a $350k property with $2000/mo rent, $4800 annual tax, and $1800 annual insurance.",
        expected_keywords=["cap rate", "NOI"],
        expected_sources=["FinancialAgent"],
    ),
    EvalCase(
        name="census_comparison",
        query="Compare the demographics of ZIP 78641 (Leander) vs 78636 (Round Mountain) — population, income, home values.",
        expected_keywords=["population", "income", "median"],
        expected_sources=["NeighborhoodAgent"],
    ),
    EvalCase(
        name="school_ratings",
        query="What are the school ratings and education quality near 78641?",
        expected_keywords=["school", "education"],
        expected_sources=["NeighborhoodAgent"],
        metadata={"address": "Leander, TX 78641"},
    ),
    EvalCase(
        name="build_vs_buy",
        query="Should I build on my vacant 1.1-acre lot in Crystal Falls (Leander TX) or buy an existing home in the $500-600k range? Consider current rates and rental potential.",
        expected_keywords=["build", "buy", "rate", "cost"],
        expected_sources=["FinancialAgent", "MarketAgent"],
        metadata={"address": "1400 Lions Lair, Leander, TX 78641"},
    ),
]


async def main():
    print("Initializing...")
    db.init_db()
    client = anthropic.AsyncAnthropic()
    coordinator, registry, logger = build_coordinator(client)

    print(f"Registry: {len(registry)} tools")
    print(f"Running {len(EVAL_CASES)} eval cases...\n")

    harness = EvalHarness(run_fn=coordinator.run)
    harness.add_cases(EVAL_CASES)

    report = await harness.run()
    print(report.summary())

    # Summary stats
    passed = sum(1 for r in report.results if r.passed)
    avg_score = sum(r.score for r in report.results) / len(report.results)
    print(f"\nTarget: 8/10 pass — Actual: {passed}/10")
    print(f"Average score: {avg_score:.0%}")


if __name__ == "__main__":
    asyncio.run(main())
