# agentkit — Build Guide for Claude Code

## What We're Building

A **reusable Python framework** for multi-agent AI systems, proven out with a **real estate property intelligence demo** as the reference application.

The framework extracts and generalizes the patterns built in `GenomAI_POC` — coordinator/specialist architecture, tool registries, grounding evaluation, eval harnesses — into a domain-agnostic library anyone can `pip install` and use to build their own multi-agent app in a new domain.

The real estate demo is not a toy. It answers real questions like:
- "Is 123 Oak Street a good buy at $425k?"
- "Compare this lot in Blanco County to comparable land sales in the last 12 months."
- "What would my cash flow look like at 20% down with today's rates?"

---

## Technology Stack

| Component | Technology |
|-----------|-----------|
| Framework core | Python 3.12, async/await |
| API server | FastAPI (same as GenomAI) |
| LLM | Claude Sonnet 4 via Anthropic API |
| Database | SQLite (dev) / PostgreSQL (optional prod) — no ClickHouse needed for this domain |
| PDF extraction | PyMuPDF |
| MCP server | FastMCP (simpler than hand-rolling as in GenomAI) |
| Frontend | Vanilla HTML/CSS/JS single-page app |
| Package build | pyproject.toml + setuptools |
| Testing | pytest + pytest-asyncio |

---

## Project Structure

```
agentkit/
├── CLAUDE.md                        ← this file
├── pyproject.toml                   ← package metadata, deps
├── README.md
│
├── agentkit/                        ← THE FRAMEWORK (pip-installable)
│   ├── __init__.py                  ← public API exports
│   ├── base.py                      ← BaseAgent: async tool-use loop
│   ├── coordinator.py               ← Coordinator: orchestrate N agents
│   ├── registry.py                  ← ToolRegistry: register + discover tools
│   ├── messages.py                  ← AgentMessage, AgentContext typed models
│   ├── grounding.py                 ← GroundingEvaluator: citation checker
│   ├── eval.py                      ← EvalHarness: structured accuracy testing
│   ├── mcp.py                       ← MCPServerBuilder: auto-generate MCP server
│   ├── logger.py                    ← InteractionLogger: audit all LLM calls
│   └── exceptions.py                ← AgentError, ToolError, GroundingError
│
├── demo/                            ← REAL ESTATE DEMO APP
│   ├── main.py                      ← FastAPI app
│   ├── config.py                    ← Pydantic settings (.env)
│   ├── start.sh
│   │
│   ├── agents/
│   │   ├── coordinator.py           ← RealEstateCoordinator
│   │   ├── market.py                ← MarketAgent: listings, comps, price trends
│   │   ├── property.py              ← PropertyAgent: parcel, tax, zoning, history
│   │   ├── neighborhood.py          ← NeighborhoodAgent: schools, walkability
│   │   ├── financial.py             ← FinancialAgent: ROI, cap rate, cash flow
│   │   └── document.py             ← DocumentAgent: deed/inspection PDF parsing
│   │
│   ├── routers/
│   │   ├── query.py                 ← POST /query (multi-agent), POST /retrieve
│   │   ├── documents.py             ← PDF upload/list
│   │   ├── properties.py            ← saved property lookups
│   │   └── ui_api.py                ← dashboard data (sources, costs, log)
│   │
│   ├── ingest/
│   │   ├── rentcast.py              ← Rentcast API (listings, comps)
│   │   ├── attom.py                 ← ATTOM API (parcel, tax, ownership)
│   │   ├── walkscore.py             ← Walk Score API
│   │   ├── fred.py                  ← FRED API (mortgage rates, economic)
│   │   └── census.py                ← US Census (demographics, income)
│   │
│   ├── db/
│   │   ├── schema.sql               ← SQLite/Postgres DDL
│   │   └── client.py                ← async DB client
│   │
│   ├── models/
│   │   └── schemas.py               ← Pydantic request/response models
│   │
│   └── static/
│       └── index.html               ← Dashboard SPA
│
├── tests/
│   ├── test_framework.py            ← agentkit unit tests (no LLM required)
│   ├── test_agents.py               ← demo agent integration tests
│   └── eval_suite.py                ← 10-query accuracy eval
│
└── examples/
    └── quickstart.py                ← 20-line "hello world" using agentkit
```

---

## Phase 1: The Framework (agentkit core)

Build this first. It should have zero dependencies on the real estate domain.

### 1.1 `agentkit/messages.py`

Typed inter-agent message models. Everything agents pass to each other.

```python
from pydantic import BaseModel, Field
from typing import Any, Optional
from enum import Enum

class AgentRole(str, Enum):
    COORDINATOR = "coordinator"
    SPECIALIST = "specialist"
    EVALUATOR = "evaluator"

class AgentMessage(BaseModel):
    """A message sent between agents."""
    sender: str                          # agent name
    recipient: str                       # agent name or "coordinator"
    content: dict[str, Any]             # payload — untyped intentionally
    message_id: str = Field(default_factory=lambda: ...)
    parent_id: Optional[str] = None     # for tracing conversation threads
    metadata: dict[str, Any] = {}

class AgentContext(BaseModel):
    """Shared context passed into every agent call."""
    session_id: str
    user_query: str
    history: list[AgentMessage] = []
    metadata: dict[str, Any] = {}       # domain-specific extras (e.g. patient_id, property_address)

class AgentResult(BaseModel):
    """What a specialist agent returns."""
    agent_name: str
    content: dict[str, Any]
    sources: list[dict] = []            # chunks/records used
    confidence: Optional[float] = None
    coverage_notes: str = ""
    tokens_used: int = 0
    cost_usd: float = 0.0
```

### 1.2 `agentkit/registry.py`

A tool registry that agents can introspect at runtime.

```python
from typing import Callable, Any
from dataclasses import dataclass, field
import inspect

@dataclass
class ToolSpec:
    name: str
    fn: Callable
    description: str
    input_schema: dict        # JSON Schema
    output_schema: dict = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)   # e.g. ["read", "expensive", "external"]

class ToolRegistry:
    """Register tools and expose them to agents."""

    def __init__(self):
        self._tools: dict[str, ToolSpec] = {}

    def register(self, name: str, description: str, tags: list[str] = None):
        """Decorator: @registry.register('my_tool', 'Does X')"""
        def decorator(fn: Callable) -> Callable:
            # auto-build input_schema from type hints
            schema = _schema_from_hints(fn)
            spec = ToolSpec(
                name=name, fn=fn, description=description,
                input_schema=schema, tags=tags or []
            )
            self._tools[name] = spec
            return fn
        return decorator

    def get(self, name: str) -> ToolSpec:
        return self._tools[name]

    def list(self, tags: list[str] = None) -> list[ToolSpec]:
        if not tags:
            return list(self._tools.values())
        return [t for t in self._tools.values() if any(tag in t.tags for tag in tags)]

    def to_anthropic_tools(self, names: list[str] = None) -> list[dict]:
        """Return tool specs in Anthropic API format."""
        specs = [self._tools[n] for n in names] if names else list(self._tools.values())
        return [
            {
                "name": s.name,
                "description": s.description,
                "input_schema": s.input_schema,
            }
            for s in specs
        ]
```

### 1.3 `agentkit/base.py`

The BaseAgent. Every specialist inherits from this. This is the main thing to get right.

Key behaviors:
- Async tool-use loop (handles multi-turn tool calling)
- Max rounds protection (default 5)
- Automatic retry with exponential backoff on API errors
- Usage tracking (tokens, cost)
- Structured logging via InteractionLogger

```python
import anthropic
import asyncio
from abc import ABC, abstractmethod
from .messages import AgentContext, AgentResult
from .registry import ToolRegistry
from .logger import InteractionLogger

class BaseAgent(ABC):
    """
    Abstract base for all specialist agents.

    Subclass this and implement:
      - name: str
      - tools: list[str]  (names in the registry)
      - system_prompt: str
      - build_user_message(context) -> str
      - parse_result(raw_response, tool_results) -> AgentResult
    """

    MODEL = "claude-sonnet-4-20250514"
    MAX_ROUNDS = 5
    MAX_RETRIES = 3

    def __init__(self, registry: ToolRegistry, logger: InteractionLogger, client: anthropic.AsyncAnthropic):
        self.registry = registry
        self.logger = logger
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
    async def parse_result(self, messages: list[dict], context: AgentContext) -> AgentResult: ...

    async def run(self, context: AgentContext) -> AgentResult:
        """Main entry point. Runs the async tool-use loop."""
        messages = [{"role": "user", "content": self.build_user_message(context)}]
        tools = self.registry.to_anthropic_tools(self.tool_names)

        for round_num in range(self.MAX_ROUNDS):
            response = await self._call_api(messages, tools)
            self.logger.log_llm_call(self.name, messages, response)

            # Check stop condition
            if response.stop_reason == "end_turn":
                messages.append({"role": "assistant", "content": response.content})
                break

            if response.stop_reason == "tool_use":
                messages.append({"role": "assistant", "content": response.content})
                # Execute all tool calls in parallel
                tool_results = await self._execute_tools(response.content)
                messages.append({"role": "user", "content": tool_results})
                continue

        return await self.parse_result(messages, context)

    async def _call_api(self, messages, tools, retries=0):
        try:
            return await self.client.messages.create(
                model=self.MODEL,
                max_tokens=4096,
                system=self.system_prompt,
                tools=tools,
                messages=messages,
            )
        except anthropic.RateLimitError:
            if retries >= self.MAX_RETRIES:
                raise
            wait = 2 ** retries
            await asyncio.sleep(wait)
            return await self._call_api(messages, tools, retries + 1)

    async def _execute_tools(self, content_blocks) -> list[dict]:
        """Execute all tool_use blocks in parallel, return tool_result blocks."""
        calls = [b for b in content_blocks if b.type == "tool_use"]
        results = await asyncio.gather(*[self._execute_one_tool(c) for c in calls])
        return [
            {"type": "tool_result", "tool_use_id": call.id, "content": str(result)}
            for call, result in zip(calls, results)
        ]

    async def _execute_one_tool(self, tool_use_block):
        spec = self.registry.get(tool_use_block.name)
        try:
            if asyncio.iscoroutinefunction(spec.fn):
                return await spec.fn(**tool_use_block.input)
            else:
                return spec.fn(**tool_use_block.input)
        except Exception as e:
            return {"error": str(e), "tool": tool_use_block.name}
```

### 1.4 `agentkit/coordinator.py`

Orchestrates specialists. Supports parallel and sequential execution patterns.

```python
import asyncio
from .messages import AgentContext, AgentResult
from .base import BaseAgent

class Coordinator:
    """
    Orchestrates multiple specialist agents.

    Usage:
        coordinator = Coordinator()
        coordinator.register_parallel(market_agent, property_agent, neighborhood_agent)
        coordinator.register_sequential(financial_agent)  # runs after parallel group
        coordinator.set_synthesizer(synthesis_fn)
        result = await coordinator.run(context)
    """

    def __init__(self):
        self._parallel_agents: list[BaseAgent] = []
        self._sequential_agents: list[BaseAgent] = []
        self._synthesizer = None

    def register_parallel(self, *agents: BaseAgent):
        self._parallel_agents.extend(agents)

    def register_sequential(self, *agents: BaseAgent):
        self._sequential_agents.extend(agents)

    def set_synthesizer(self, fn):
        """fn(context, parallel_results, sequential_results) -> str"""
        self._synthesizer = fn

    async def run(self, context: AgentContext) -> dict:
        # Phase 1: parallel specialists
        parallel_results: list[AgentResult] = await asyncio.gather(
            *[agent.run(context) for agent in self._parallel_agents],
            return_exceptions=True
        )
        # Filter exceptions, log them
        parallel_results = [r for r in parallel_results if isinstance(r, AgentResult)]

        # Phase 2: sequential specialists (each can see prior results)
        sequential_results = []
        for agent in self._sequential_agents:
            context.metadata["prior_results"] = parallel_results + sequential_results
            result = await agent.run(context)
            sequential_results.append(result)

        # Phase 3: synthesize
        final_answer = ""
        if self._synthesizer:
            final_answer = await self._synthesizer(context, parallel_results, sequential_results)

        return {
            "answer": final_answer,
            "agent_results": parallel_results + sequential_results,
            "total_cost": sum(r.cost_usd for r in parallel_results + sequential_results),
        }
```

### 1.5 `agentkit/grounding.py`

Evaluates how well the answer is supported by the source chunks. Reuse the pattern from GenomAI's Clinical Reasoning Agent.

```python
import anthropic
from pydantic import BaseModel

class GroundingClaim(BaseModel):
    claim: str
    supported: bool
    evidence: str        # quote from sources, or "not found"
    confidence: float    # 0.0-1.0

class GroundingResult(BaseModel):
    score: float                    # 0.0-1.0
    claims: list[GroundingClaim]
    summary: str

class GroundingEvaluator:
    """
    Send answer + source chunks to Claude and get back a grounding score.
    Same pattern as GenomAI's inline grounding check.
    """

    def __init__(self, client: anthropic.AsyncAnthropic):
        self.client = client

    async def evaluate(self, answer: str, sources: list[dict]) -> GroundingResult:
        source_text = "\n\n".join(
            f"[SOURCE {i}]\n{s.get('text', str(s))}" for i, s in enumerate(sources)
        )
        prompt = f"""
You are evaluating whether an AI answer is grounded in the provided source data.

ANSWER:
{answer}

SOURCES:
{source_text}

For each factual claim in the answer, determine whether it is directly supported by the sources.
Respond as JSON: {{ "score": 0.0-1.0, "claims": [...], "summary": "..." }}
Each claim: {{ "claim": "...", "supported": true/false, "evidence": "...", "confidence": 0.0-1.0 }}
"""
        response = await self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
        )
        import json
        raw = response.content[0].text
        data = json.loads(raw)
        return GroundingResult(**data)
```

### 1.6 `agentkit/eval.py`

A structured test harness for evaluating agent pipelines. Returns a report you can save and compare across versions.

```python
from dataclasses import dataclass
from typing import Callable, Any
import time

@dataclass
class EvalCase:
    name: str
    query: str
    expected_keywords: list[str]         # must appear in answer
    expected_sources: list[str]          # agent names that must contribute
    metadata: dict = None                # e.g. {"address": "123 Oak St"}

@dataclass
class EvalResult:
    case_name: str
    passed: bool
    score: float                         # 0.0-1.0
    missing_keywords: list[str]
    missing_sources: list[str]
    latency_s: float
    cost_usd: float
    answer_snippet: str

class EvalHarness:
    """
    Run a suite of EvalCases against an agent pipeline and report accuracy.

    Usage:
        harness = EvalHarness(run_fn=coordinator.run)
        harness.add_cases(REAL_ESTATE_EVAL_CASES)
        report = await harness.run()
        print(report.summary())
    """

    def __init__(self, run_fn: Callable):
        self.run_fn = run_fn
        self.cases: list[EvalCase] = []

    def add_cases(self, cases: list[EvalCase]):
        self.cases.extend(cases)

    async def run_case(self, case: EvalCase) -> EvalResult:
        from .messages import AgentContext
        import uuid
        ctx = AgentContext(
            session_id=str(uuid.uuid4()),
            user_query=case.query,
            metadata=case.metadata or {}
        )
        t0 = time.monotonic()
        result = await self.run_fn(ctx)
        latency = time.monotonic() - t0

        answer = result.get("answer", "")
        agent_names = [r.agent_name for r in result.get("agent_results", [])]

        missing_kw = [kw for kw in case.expected_keywords if kw.lower() not in answer.lower()]
        missing_src = [s for s in case.expected_sources if s not in agent_names]

        score = 1.0 - (len(missing_kw) / max(len(case.expected_keywords), 1)) * 0.6 \
                    - (len(missing_src) / max(len(case.expected_sources), 1)) * 0.4

        return EvalResult(
            case_name=case.name,
            passed=(score >= 0.7),
            score=score,
            missing_keywords=missing_kw,
            missing_sources=missing_src,
            latency_s=latency,
            cost_usd=result.get("total_cost", 0.0),
            answer_snippet=answer[:200],
        )

    async def run(self) -> "EvalReport":
        import asyncio
        results = await asyncio.gather(*[self.run_case(c) for c in self.cases])
        return EvalReport(results=list(results))

@dataclass
class EvalReport:
    results: list[EvalResult]

    def summary(self) -> str:
        passed = sum(1 for r in self.results if r.passed)
        total_cost = sum(r.cost_usd for r in self.results)
        avg_latency = sum(r.latency_s for r in self.results) / len(self.results)
        lines = [
            f"\nEval Results: {passed}/{len(self.results)} passed",
            f"Total cost: ${total_cost:.4f} | Avg latency: {avg_latency:.1f}s",
            "-" * 50
        ]
        for r in self.results:
            status = "✓" if r.passed else "✗"
            lines.append(f"{status} [{r.score:.0%}] {r.case_name}")
            if r.missing_keywords:
                lines.append(f"   Missing keywords: {r.missing_keywords}")
        return "\n".join(lines)
```

### 1.7 `agentkit/mcp.py`

Auto-generate an MCP server from a Coordinator. Use FastMCP (much simpler than hand-rolling).

```python
from fastmcp import FastMCP

class MCPServerBuilder:
    """
    Build a FastMCP server from an agentkit Coordinator.
    
    Usage:
        builder = MCPServerBuilder(coordinator, name="real-estate")
        mcp = builder.build()
        mcp.run()  # or: mcp.run(transport="sse", port=8001)
    """

    def __init__(self, coordinator, name: str):
        self.coordinator = coordinator
        self.name = name

    def build(self) -> FastMCP:
        mcp = FastMCP(self.name)
        coordinator = self.coordinator

        @mcp.tool()
        async def query_property(
            question: str,
            property_address: str = "",
            session_id: str = ""
        ) -> str:
            """Ask a question about a property or real estate market."""
            from .messages import AgentContext
            import uuid
            ctx = AgentContext(
                session_id=session_id or str(uuid.uuid4()),
                user_query=question,
                metadata={"address": property_address}
            )
            result = await coordinator.run(ctx)
            return result.get("answer", "No answer generated.")

        return mcp
```

---

## Phase 2: Real Estate Demo Agents

Now build the 5 specialist agents using the framework.

### Agent 2.1: MarketAgent (`demo/agents/market.py`)

Answers: current listings near an address, comparable sales, price trends, days on market.

**Tools to register:**
- `search_listings(address, radius_miles, max_results)` → Rentcast `/v1/listings/sale`
- `get_comps(address, radius_miles, sold_in_days)` → Rentcast `/v1/avm/sale` comparable sales
- `get_market_stats(zipcode)` → Rentcast market statistics

**System prompt focus:** "You are a real estate market analyst. Use listing and comparable sale data to characterize market conditions. Always cite specific data points (price per sq ft, days on market, list-to-sale ratio)."

### Agent 2.2: PropertyAgent (`demo/agents/property.py`)

Answers: parcel details, tax history, ownership chain, lot characteristics, zoning.

**Tools to register:**
- `get_parcel(address)` → ATTOM `/v1/property/basicprofile`
- `get_tax_history(attom_id)` → ATTOM `/v1/assessment/history`
- `get_sale_history(attom_id)` → ATTOM `/v1/saleshistory/detail`
- `get_avm(address)` → ATTOM automated valuation model

**System prompt focus:** "You are a property records analyst. Report factual data about parcel characteristics, tax assessments, and ownership history. Note any unusual patterns in tax history or ownership transfers."

### Agent 2.3: NeighborhoodAgent (`demo/agents/neighborhood.py`)

Answers: walk/transit/bike scores, nearby schools (ratings), flood zone, census demographics.

**Tools to register:**
- `get_walk_score(address, lat, lon)` → Walk Score API
- `get_school_ratings(lat, lon, radius)` → GreatSchools API or NCES (free)
- `get_demographics(zipcode)` → US Census ACS 5-year estimates
- `get_flood_zone(lat, lon)` → FEMA National Flood Hazard Layer (free, GeoJSON)

**System prompt focus:** "You are a neighborhood research analyst. Provide objective scores and data about livability, school quality, commute options, and environmental risks."

### Agent 2.4: FinancialAgent (`demo/agents/financial.py`)

Answers: estimated cash flow, cap rate, ROI, break-even, mortgage payment at current rates.

This agent is **mostly computational** — minimal LLM, mostly Python math. The tools do the heavy lifting; the LLM formats the output and explains the numbers.

**Tools to register:**
- `get_current_mortgage_rates()` → FRED API, series MORTGAGE30US
- `calculate_mortgage(price, down_pct, rate, term_years)` → pure Python
- `estimate_rental_income(address, bedrooms, bathrooms)` → Rentcast `/v1/avm/rent/long-term`
- `calculate_cash_flow(purchase_price, down_pct, rate, monthly_rent, tax_annual, insurance_annual, vacancy_rate, management_pct)` → pure Python
- `calculate_cap_rate(noi, purchase_price)` → pure Python

**System prompt focus:** "You are a real estate investment analyst. Show all calculations clearly. State your assumptions. Flag when key inputs (rent estimate, tax, insurance) are estimated vs. known."

### Agent 2.5: DocumentAgent (`demo/agents/document.py`)

Parses uploaded PDFs (deed, inspection report, HOA docs, appraisal).

**Tools to register:**
- `extract_deed_data(doc_id)` → reads from `documents` table, returns extracted fields
- `extract_inspection_issues(doc_id)` → uses PyMuPDF + Claude to extract issues by severity
- `list_documents(address)` → returns uploaded docs for a property

**System prompt focus:** "You are a real estate document analyst. Extract factual information from property documents. For inspection reports, categorize issues as major/minor/cosmetic and flag anything that affects safety or habitability. Always cite page numbers."

### Coordinator (`demo/agents/coordinator.py`)

```
User question
     │
     ▼
RealEstateCoordinator
     │
     ├──── [parallel] MarketAgent
     ├──── [parallel] PropertyAgent
     ├──── [parallel] NeighborhoodAgent
     ├──── [parallel] DocumentAgent (if docs uploaded)
     │
     └──── [sequential] FinancialAgent (needs market + property data)
                │
                ▼
          Synthesis (Claude call: combine all results into coherent answer)
                │
                ▼
          GroundingEvaluator (inline grounding check)
```

The coordinator's synthesis prompt should:
1. Receive the structured output from all 4-5 agents
2. Produce a coherent answer in 3-5 paragraphs
3. Include a "Bottom line" section with a clear recommendation
4. Cite specific data from each agent

---

## Phase 3: FastAPI Service + Dashboard

### `demo/main.py` — Key endpoints

```
POST /query              ← multi-agent: returns answer + agent trace + grounding score
POST /retrieve           ← retrieval only, no LLM synthesis
POST /documents/upload   ← PDF upload (PyMuPDF extraction)
GET  /documents          ← list docs (filterable by address)
GET  /properties         ← saved property lookups
GET  /api/sources        ← data source status (API health)
GET  /api/session-cost   ← token usage + cost tracker
GET  /api/interaction-log← LLM call audit trail
```

### Dashboard SPA (`demo/static/index.html`)

Copy the GenomAI dashboard structure but adapt to real estate:

- **Address bar** — enter a property address; this is the "active property"
- **Query console** — natural language Q&A about the active property
- **Agent trace** — expandable view of which agents ran + what tools they called
- **Grounding badge** — green/yellow/red, click to expand claim-by-claim
- **Cost tracker** — last query cost, session total
- **Document drop zone** — drag-and-drop PDF upload for the active property
- **Data source status** — API health (Rentcast, ATTOM, Walk Score, FRED)
- **Saved properties** — list of previously analyzed properties

---

## Phase 4: Package + MCP Server

### `pyproject.toml`

```toml
[project]
name = "agentkit"
version = "0.1.0"
description = "Reusable multi-agent framework for AI applications"
requires-python = ">=3.12"
dependencies = [
    "anthropic>=0.40.0",
    "fastapi>=0.115.0",
    "pydantic>=2.0.0",
    "fastmcp>=0.1.0",
    "httpx>=0.27.0",
    "pymupdf>=1.24.0",
    "uvicorn>=0.30.0",
    "python-dotenv>=1.0.0",
]

[project.optional-dependencies]
dev = ["pytest", "pytest-asyncio", "httpx"]

[project.scripts]
agentkit-mcp = "agentkit.mcp:run_server"
```

### MCP Server

The MCP server wraps the coordinator and exposes it to Claude Desktop.

Add to `~/.claude/claude_desktop_config.json`:
```json
{
  "mcpServers": {
    "real-estate": {
      "command": "python",
      "args": ["-m", "agentkit.mcp"],
      "env": {
        "RENTCAST_API_KEY": "...",
        "ATTOM_API_KEY": "...",
        "WALKSCORE_API_KEY": "..."
      }
    }
  }
}
```

---

## Phase 5: Eval Suite

### `tests/eval_suite.py` — 10 test cases

```python
EVAL_CASES = [
    EvalCase(
        name="basic_valuation",
        query="Is 2804 Ridgewood Dr, Austin TX 78723 fairly priced at $489k?",
        expected_keywords=["comparable", "price per sq ft", "market"],
        expected_sources=["MarketAgent", "PropertyAgent"],
    ),
    EvalCase(
        name="investment_cashflow",
        query="What would my monthly cash flow be on a $350k rental in Leander TX at 20% down?",
        expected_keywords=["cash flow", "mortgage", "rental income", "cap rate"],
        expected_sources=["FinancialAgent", "MarketAgent"],
    ),
    EvalCase(
        name="neighborhood_schools",
        query="What are the school ratings near 78641?",
        expected_keywords=["school", "rating", "district"],
        expected_sources=["NeighborhoodAgent"],
    ),
    EvalCase(
        name="land_valuation",
        query="Compare land prices per acre in Blanco County TX vs Burnet County TX",
        expected_keywords=["acre", "Blanco", "Burnet", "comparable"],
        expected_sources=["MarketAgent", "PropertyAgent"],
    ),
    EvalCase(
        name="rate_impact",
        query="How does today's 30-year rate affect affordability on a $400k home vs 12 months ago?",
        expected_keywords=["rate", "payment", "affordability"],
        expected_sources=["FinancialAgent"],
    ),
    # ... 5 more covering: flood risk, tax history, HOA doc parsing, cap rate comparison, market trends
]
```

Target: 8/10 pass, average grounding ≥ 75%.

---

## Environment Variables

```bash
# .env
ANTHROPIC_API_KEY=...

# Data APIs (all have free tiers)
RENTCAST_API_KEY=...           # rentcast.io — listings, comps, rent estimates
ATTOM_API_KEY=...              # attomdata.com — parcel, tax, AVM
WALKSCORE_API_KEY=...          # walkscore.com — walk/transit/bike
# FRED is free, no key required for public series
# US Census is free, no key required for ACS estimates
# FEMA flood data is free GeoJSON

LOG_LEVEL=INFO
DB_URL=sqlite:///./agentkit_demo.db
```

---

## Build Order for Claude Code

Follow this sequence exactly:

1. **`agentkit/` core** — messages, registry, base, coordinator, grounding, eval, mcp, logger (no external deps, fully testable)
2. **`tests/test_framework.py`** — unit tests for core with mocks (no LLM calls)
3. **`demo/ingest/`** — API client wrappers for each data source
4. **`demo/agents/`** — 5 specialist agents + coordinator
5. **`demo/routers/`** + `demo/main.py` — FastAPI service
6. **`demo/static/index.html`** — dashboard SPA
7. **`pyproject.toml`** + `examples/quickstart.py`
8. **`tests/eval_suite.py`** — run evals, iterate on prompts until 8/10 pass
9. **`agentkit/mcp.py`** + MCP server wiring

---

## Key Design Decisions

### Why SQLite not ClickHouse?
Real estate data is low volume — a few thousand property records. SQLite is zero-ops, ships with Python, and teaches the same ORM/async patterns. ClickHouse would be overkill and adds setup friction.

### Why FastMCP not hand-rolled MCP?
GenomAI hand-rolled its MCP server as a learning exercise. Now that you understand the protocol, use FastMCP to show the production pattern. The `MCPServerBuilder` class in `agentkit/mcp.py` becomes a reusable piece — anyone using the framework gets MCP support for free.

### Why 5 agents not one?
The specialization is the lesson. Each agent has a focused system prompt, a focused tool set, and limited context. The coordinator composes them. This pattern scales: adding a new data source means adding a new agent, not modifying an existing one.

### What makes this a "framework" vs just a demo?
The `agentkit/` package must be usable without the `demo/`. The `examples/quickstart.py` should show how to build a completely different multi-agent app (e.g. a weather analyst, a stock screener) using only the framework primitives.

---

## Free API Tiers (No CC Required for Dev)

| API | Free Tier | Needed For |
|-----|-----------|-----------|
| Rentcast | 50 req/month | Listings, comps, rent estimates |
| ATTOM | Trial available | Parcel, tax, AVM |
| Walk Score | 5,000 req/day | Walkability scores |
| FRED (St. Louis Fed) | Unlimited | Mortgage rates, economic data |
| US Census ACS | Unlimited | Demographics, income |
| FEMA Flood | Unlimited (GeoJSON) | Flood zone risk |

Start with FRED (no signup) and Walk Score (instant approval) to unblock the FinancialAgent and NeighborhoodAgent while waiting for Rentcast/ATTOM trial access.

---

## What You'll Know When Done

After building this you'll have hands-on experience with:
- **Agent lifecycle management** — async tool-use loops, max rounds, retry logic
- **Parallel agent orchestration** — asyncio.gather, fan-out/fan-in patterns
- **Tool registries** — decoupled tool discovery, auto-schema generation
- **Grounding evaluation** — measuring hallucination vs. source support
- **Eval-driven development** — write the eval suite first, iterate until it passes
- **Python packaging** — pyproject.toml, editable installs, entry points
- **MCP server publishing** — expose any agent pipeline to Claude Desktop
- **FastMCP** — production MCP pattern (vs. the GenomAI hand-rolled approach)

This framework is directly applicable to Castellan — specifically, the `Coordinator`, `BaseAgent`, `ToolRegistry`, and `EvalHarness` classes are the kind of primitives that belong in an agent governance platform.
