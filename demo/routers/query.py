"""Query router — multi-agent Q&A and retrieval-only endpoints."""

import time
import uuid

import anthropic
from fastapi import APIRouter, HTTPException

from agentkit import AgentContext
from demo.agents.coordinator import build_coordinator, run_with_grounding
from demo.db import client as db
from demo.models.schemas import QueryRequest, QueryResponse

router = APIRouter(tags=["query"])

# Build once at import time; shared across requests
_client = anthropic.AsyncAnthropic()
_coordinator, _registry, _logger = build_coordinator(_client)


@router.post("/query", response_model=QueryResponse)
async def multi_agent_query(req: QueryRequest):
    """Run the full multi-agent pipeline: specialists → synthesis → grounding."""
    session_id = req.session_id or uuid.uuid4().hex
    context = AgentContext(
        session_id=session_id,
        user_query=req.question,
        metadata={"address": req.address} if req.address else {},
    )

    t0 = time.monotonic()
    try:
        result = await run_with_grounding(_coordinator, context, _client)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    latency = time.monotonic() - t0

    answer = result.get("answer", "")
    grounding_score = result.get("grounding_score", 0.0)
    agent_results = result.get("agent_results", [])
    total_cost = result.get("total_cost", 0.0)
    total_tokens = result.get("total_tokens", 0)

    agent_trace = [
        {
            "agent": r.agent_name,
            "confidence": r.confidence,
            "sources_count": len(r.sources),
            "tokens": r.tokens_used,
            "cost": r.cost_usd,
            "snippet": r.content.get("analysis", "")[:300],
        }
        for r in agent_results
    ]

    # Persist to DB
    try:
        db.save_query(
            session_id=session_id,
            user_query=req.question,
            answer=answer,
            grounding_score=grounding_score,
            agent_trace=agent_trace,
            total_cost=total_cost,
            latency_s=latency,
        )
    except Exception:
        pass  # Don't fail the request if logging fails

    return QueryResponse(
        answer=answer,
        grounding_score=grounding_score,
        agent_trace=agent_trace,
        total_cost=total_cost,
        total_tokens=total_tokens,
        session_id=session_id,
    )


@router.post("/retrieve")
async def retrieve_only(req: QueryRequest):
    """Retrieval only — run tools without LLM synthesis. Returns raw agent data."""
    session_id = req.session_id or uuid.uuid4().hex
    context = AgentContext(
        session_id=session_id,
        user_query=req.question,
        metadata={"address": req.address} if req.address else {},
    )

    try:
        result = await _coordinator.run(context)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "agent_results": [
            {
                "agent": r.agent_name,
                "content": r.content,
                "sources": r.sources,
                "confidence": r.confidence,
            }
            for r in result.get("agent_results", [])
        ],
        "session_id": session_id,
    }
