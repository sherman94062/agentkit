"""Unit tests for agentkit core — no LLM calls required."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from agentkit import (
    AgentContext,
    AgentMessage,
    AgentResult,
    AgentRole,
    BaseAgent,
    Coordinator,
    EvalCase,
    EvalHarness,
    EvalReport,
    EvalResult,
    InteractionLogger,
    ToolError,
    ToolRegistry,
    ToolSpec,
)


# ---------------------------------------------------------------------------
# messages.py
# ---------------------------------------------------------------------------


class TestMessages:
    def test_agent_context_defaults(self):
        ctx = AgentContext(session_id="s1", user_query="hello")
        assert ctx.history == []
        assert ctx.metadata == {}

    def test_agent_message_id_generated(self):
        msg = AgentMessage(sender="a", recipient="b", content={"key": "val"})
        assert len(msg.message_id) > 0

    def test_agent_result_defaults(self):
        r = AgentResult(agent_name="test", content={"answer": "hi"})
        assert r.sources == []
        assert r.confidence is None
        assert r.tokens_used == 0
        assert r.cost_usd == 0.0

    def test_agent_role_values(self):
        assert AgentRole.COORDINATOR == "coordinator"
        assert AgentRole.SPECIALIST == "specialist"


# ---------------------------------------------------------------------------
# registry.py
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_register_decorator(self):
        reg = ToolRegistry()

        @reg.register("add", "Add two numbers")
        def add(a: int, b: int) -> int:
            return a + b

        assert "add" in reg
        assert len(reg) == 1
        spec = reg.get("add")
        assert spec.name == "add"
        assert spec.fn(2, 3) == 5

    def test_register_with_tags(self):
        reg = ToolRegistry()

        @reg.register("fetch", "Fetch data", tags=["external", "read"])
        def fetch(url: str) -> str:
            return url

        specs = reg.list(tags=["external"])
        assert len(specs) == 1
        assert specs[0].name == "fetch"

    def test_list_no_tags_returns_all(self):
        reg = ToolRegistry()
        reg.add("a", lambda: 1, "tool a")
        reg.add("b", lambda: 2, "tool b")
        assert len(reg.list()) == 2

    def test_get_missing_raises(self):
        reg = ToolRegistry()
        with pytest.raises(ToolError):
            reg.get("nonexistent")

    def test_imperative_add(self):
        reg = ToolRegistry()
        reg.add("greet", lambda name: f"hi {name}", "Greet someone")
        assert reg.get("greet").fn("world") == "hi world"

    def test_to_anthropic_tools_format(self):
        reg = ToolRegistry()

        @reg.register("echo", "Echo input")
        def echo(text: str) -> str:
            return text

        tools = reg.to_anthropic_tools()
        assert len(tools) == 1
        assert tools[0]["name"] == "echo"
        assert "input_schema" in tools[0]
        assert tools[0]["input_schema"]["type"] == "object"
        assert "text" in tools[0]["input_schema"]["properties"]

    def test_to_anthropic_tools_filter_by_name(self):
        reg = ToolRegistry()
        reg.add("a", lambda: 1, "a")
        reg.add("b", lambda: 2, "b")
        reg.add("c", lambda: 3, "c")
        tools = reg.to_anthropic_tools(names=["a", "c"])
        assert len(tools) == 2
        names = {t["name"] for t in tools}
        assert names == {"a", "c"}

    def test_schema_from_hints_defaults(self):
        reg = ToolRegistry()

        @reg.register("search", "Search")
        def search(query: str, limit: int = 10) -> list:
            return []

        schema = reg.get("search").input_schema
        assert "query" in schema["required"]
        assert "limit" not in schema["required"]


# ---------------------------------------------------------------------------
# logger.py
# ---------------------------------------------------------------------------


class TestLogger:
    def test_log_entry_tracking(self):
        logger = InteractionLogger()
        response = MagicMock()
        response.usage.input_tokens = 100
        response.usage.output_tokens = 50
        response.content = [MagicMock(text="hello", type="text")]
        response.model = "claude-sonnet-4-20250514"

        logger.start_call()
        entry = logger.log_llm_call("test_agent", [{"role": "user"}], response)

        assert entry.agent_name == "test_agent"
        assert entry.input_tokens == 100
        assert entry.output_tokens == 50
        assert entry.cost_usd > 0
        assert len(logger.entries) == 1

    def test_total_cost(self):
        logger = InteractionLogger()
        response = MagicMock()
        response.usage.input_tokens = 1000
        response.usage.output_tokens = 500
        response.content = [MagicMock(text="x")]
        response.model = "test"

        logger.log_llm_call("a", [], response)
        logger.log_llm_call("b", [], response)
        assert logger.total_cost > 0
        assert len(logger.entries) == 2

    def test_to_dicts(self):
        logger = InteractionLogger()
        response = MagicMock()
        response.usage.input_tokens = 10
        response.usage.output_tokens = 5
        response.content = [MagicMock(text="hi")]
        response.model = "m"

        logger.log_llm_call("agent", [{"role": "user"}], response)
        dicts = logger.to_dicts()
        assert len(dicts) == 1
        assert dicts[0]["agent"] == "agent"


# ---------------------------------------------------------------------------
# base.py — test with a mock agent
# ---------------------------------------------------------------------------


class MockAgent(BaseAgent):
    @property
    def name(self) -> str:
        return "mock_agent"

    @property
    def system_prompt(self) -> str:
        return "You are a test agent."

    @property
    def tool_names(self) -> list[str]:
        return ["add"]

    def build_user_message(self, context: AgentContext) -> str:
        return context.user_query

    async def parse_result(self, messages, context):
        last = messages[-1]
        text = ""
        if isinstance(last.get("content"), list):
            for block in last["content"]:
                if hasattr(block, "text"):
                    text = block.text
                    break
        return AgentResult(agent_name=self.name, content={"answer": text})


class TestBaseAgent:
    @pytest.mark.asyncio
    async def test_run_end_turn(self):
        reg = ToolRegistry()
        reg.add("add", lambda a, b: a + b, "Add")

        mock_response = MagicMock()
        mock_response.stop_reason = "end_turn"
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "The answer is 5"
        mock_response.content = [text_block]
        mock_response.usage.input_tokens = 50
        mock_response.usage.output_tokens = 20
        mock_response.model = "test"

        client = AsyncMock()
        client.messages.create = AsyncMock(return_value=mock_response)

        logger = InteractionLogger()
        agent = MockAgent(registry=reg, logger=logger, client=client)
        ctx = AgentContext(session_id="s1", user_query="what is 2+3?")

        result = await agent.run(ctx)
        assert result.agent_name == "mock_agent"
        assert result.content["answer"] == "The answer is 5"
        assert len(logger.entries) == 1

    @pytest.mark.asyncio
    async def test_run_tool_use_then_end(self):
        reg = ToolRegistry()
        reg.add("add", lambda a, b: a + b, "Add")

        # First response: tool_use
        tool_block = MagicMock()
        tool_block.type = "tool_use"
        tool_block.name = "add"
        tool_block.id = "tool_1"
        tool_block.input = {"a": 2, "b": 3}

        resp1 = MagicMock()
        resp1.stop_reason = "tool_use"
        resp1.content = [tool_block]
        resp1.usage.input_tokens = 30
        resp1.usage.output_tokens = 10
        resp1.model = "test"

        # Second response: end_turn
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "5"

        resp2 = MagicMock()
        resp2.stop_reason = "end_turn"
        resp2.content = [text_block]
        resp2.usage.input_tokens = 40
        resp2.usage.output_tokens = 10
        resp2.model = "test"

        client = AsyncMock()
        client.messages.create = AsyncMock(side_effect=[resp1, resp2])

        logger = InteractionLogger()
        agent = MockAgent(registry=reg, logger=logger, client=client)
        ctx = AgentContext(session_id="s1", user_query="add 2+3")

        result = await agent.run(ctx)
        assert result.content["answer"] == "5"
        assert len(logger.entries) == 2  # two LLM calls


# ---------------------------------------------------------------------------
# coordinator.py
# ---------------------------------------------------------------------------


class TestCoordinator:
    @pytest.mark.asyncio
    async def test_parallel_agents(self):
        agent1 = AsyncMock(spec=BaseAgent)
        agent1.name = "agent1"
        agent1.run = AsyncMock(
            return_value=AgentResult(
                agent_name="agent1", content={"data": "a"}, cost_usd=0.01, tokens_used=100
            )
        )
        agent2 = AsyncMock(spec=BaseAgent)
        agent2.name = "agent2"
        agent2.run = AsyncMock(
            return_value=AgentResult(
                agent_name="agent2", content={"data": "b"}, cost_usd=0.02, tokens_used=200
            )
        )

        coord = Coordinator()
        coord.register_parallel(agent1, agent2)
        coord.set_synthesizer(
            AsyncMock(return_value="Combined answer from a and b.")
        )

        ctx = AgentContext(session_id="s1", user_query="test")
        result = await coord.run(ctx)

        assert result["answer"] == "Combined answer from a and b."
        assert len(result["agent_results"]) == 2
        assert result["total_cost"] == pytest.approx(0.03)

    @pytest.mark.asyncio
    async def test_sequential_agents_see_prior(self):
        agent_p = AsyncMock(spec=BaseAgent)
        agent_p.name = "parallel"
        agent_p.run = AsyncMock(
            return_value=AgentResult(agent_name="parallel", content={"v": 1})
        )

        agent_s = AsyncMock(spec=BaseAgent)
        agent_s.name = "sequential"
        agent_s.run = AsyncMock(
            return_value=AgentResult(agent_name="sequential", content={"v": 2})
        )

        coord = Coordinator()
        coord.register_parallel(agent_p)
        coord.register_sequential(agent_s)

        ctx = AgentContext(session_id="s1", user_query="test")
        result = await coord.run(ctx)

        assert len(result["agent_results"]) == 2
        # Sequential agent should have been called with prior_results in metadata
        call_ctx = agent_s.run.call_args[0][0]
        assert "prior_results" in call_ctx.metadata

    @pytest.mark.asyncio
    async def test_failed_parallel_agent_excluded(self):
        good = AsyncMock(spec=BaseAgent)
        good.name = "good"
        good.run = AsyncMock(
            return_value=AgentResult(agent_name="good", content={"ok": True})
        )

        bad = AsyncMock(spec=BaseAgent)
        bad.name = "bad"
        bad.run = AsyncMock(side_effect=RuntimeError("boom"))

        coord = Coordinator()
        coord.register_parallel(good, bad)

        ctx = AgentContext(session_id="s1", user_query="test")
        result = await coord.run(ctx)

        assert len(result["agent_results"]) == 1
        assert result["agent_results"][0].agent_name == "good"


# ---------------------------------------------------------------------------
# eval.py
# ---------------------------------------------------------------------------


class TestEval:
    @pytest.mark.asyncio
    async def test_eval_harness_pass(self):
        async def mock_run(ctx):
            return {
                "answer": "The market shows comparable sales at $200 per sq ft.",
                "agent_results": [
                    AgentResult(agent_name="MarketAgent", content={}, cost_usd=0.01)
                ],
                "total_cost": 0.01,
            }

        harness = EvalHarness(run_fn=mock_run)
        harness.add_cases([
            EvalCase(
                name="market_test",
                query="What are comps?",
                expected_keywords=["comparable", "sq ft"],
                expected_sources=["MarketAgent"],
            )
        ])
        report = await harness.run()
        assert len(report.results) == 1
        assert report.results[0].passed is True
        assert report.results[0].score == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_eval_harness_fail(self):
        async def mock_run(ctx):
            return {
                "answer": "I don't know.",
                "agent_results": [],
                "total_cost": 0.0,
            }

        harness = EvalHarness(run_fn=mock_run)
        harness.add_cases([
            EvalCase(
                name="missing_everything",
                query="test",
                expected_keywords=["price", "market"],
                expected_sources=["MarketAgent"],
            )
        ])
        report = await harness.run()
        assert report.results[0].passed is False
        assert report.results[0].score < 0.7

    def test_eval_report_summary(self):
        results = [
            EvalResult(
                case_name="test1", passed=True, score=0.9,
                missing_keywords=[], missing_sources=[],
                latency_s=1.5, cost_usd=0.01, answer_snippet="ok"
            ),
            EvalResult(
                case_name="test2", passed=False, score=0.3,
                missing_keywords=["price"], missing_sources=["MarketAgent"],
                latency_s=2.0, cost_usd=0.02, answer_snippet="bad"
            ),
        ]
        report = EvalReport(results=results)
        summary = report.summary()
        assert "1/2 passed" in summary
        assert "test1" in summary
        assert "Missing keywords" in summary
