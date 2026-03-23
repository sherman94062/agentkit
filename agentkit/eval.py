"""Eval harness — structured accuracy testing for agent pipelines."""

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from .messages import AgentContext


@dataclass
class EvalCase:
    name: str
    query: str
    expected_keywords: list[str]
    expected_sources: list[str]
    metadata: dict[str, Any] | None = None


@dataclass
class EvalResult:
    case_name: str
    passed: bool
    score: float
    missing_keywords: list[str]
    missing_sources: list[str]
    latency_s: float
    cost_usd: float
    answer_snippet: str


@dataclass
class EvalReport:
    results: list[EvalResult]

    def summary(self) -> str:
        passed = sum(1 for r in self.results if r.passed)
        total_cost = sum(r.cost_usd for r in self.results)
        avg_latency = (
            sum(r.latency_s for r in self.results) / len(self.results)
            if self.results
            else 0
        )
        lines = [
            f"\nEval Results: {passed}/{len(self.results)} passed",
            f"Total cost: ${total_cost:.4f} | Avg latency: {avg_latency:.1f}s",
            "-" * 50,
        ]
        for r in self.results:
            status = "PASS" if r.passed else "FAIL"
            lines.append(f"{status} [{r.score:.0%}] {r.case_name}")
            if r.missing_keywords:
                lines.append(f"   Missing keywords: {r.missing_keywords}")
            if r.missing_sources:
                lines.append(f"   Missing sources: {r.missing_sources}")
        return "\n".join(lines)


class EvalHarness:
    """
    Run a suite of EvalCases against an agent pipeline and report accuracy.

    Usage:
        harness = EvalHarness(run_fn=coordinator.run)
        harness.add_cases(cases)
        report = await harness.run()
        print(report.summary())
    """

    def __init__(self, run_fn: Callable):
        self.run_fn = run_fn
        self.cases: list[EvalCase] = []

    def add_cases(self, cases: list[EvalCase]) -> None:
        self.cases.extend(cases)

    async def run_case(self, case: EvalCase) -> EvalResult:
        ctx = AgentContext(
            session_id=uuid.uuid4().hex,
            user_query=case.query,
            metadata=case.metadata or {},
        )
        t0 = time.monotonic()
        result = await self.run_fn(ctx)
        latency = time.monotonic() - t0

        answer = result.get("answer", "")
        agent_names = [r.agent_name for r in result.get("agent_results", [])]

        missing_kw = [
            kw for kw in case.expected_keywords if kw.lower() not in answer.lower()
        ]
        missing_src = [s for s in case.expected_sources if s not in agent_names]

        kw_score = 1.0 - (
            len(missing_kw) / max(len(case.expected_keywords), 1)
        )
        src_score = 1.0 - (
            len(missing_src) / max(len(case.expected_sources), 1)
        )
        score = kw_score * 0.6 + src_score * 0.4

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

    async def run(self) -> EvalReport:
        results = await asyncio.gather(*[self.run_case(c) for c in self.cases])
        return EvalReport(results=list(results))
