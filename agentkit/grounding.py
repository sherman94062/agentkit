"""Grounding evaluator — checks if answers are supported by source data."""

import json
import logging

import anthropic
from pydantic import BaseModel, Field

logger = logging.getLogger("agentkit")


class GroundingClaim(BaseModel):
    claim: str
    supported: bool
    evidence: str
    confidence: float


class GroundingResult(BaseModel):
    score: float
    claims: list[GroundingClaim] = Field(default_factory=list)
    summary: str = ""


class GroundingEvaluator:
    """
    Send answer + source chunks to Claude and get back a grounding score.
    """

    MODEL = "claude-sonnet-4-20250514"

    def __init__(self, client: anthropic.AsyncAnthropic):
        self.client = client

    async def evaluate(self, answer: str, sources: list[dict]) -> GroundingResult:
        if not sources:
            return GroundingResult(
                score=0.0, claims=[], summary="No sources provided for grounding."
            )

        source_text = "\n\n".join(
            f"[SOURCE {i}]\n{s.get('text', json.dumps(s, default=str))}"
            for i, s in enumerate(sources)
        )
        prompt = f"""You are evaluating whether an AI answer is grounded in the provided source data.

ANSWER:
{answer}

SOURCES:
{source_text}

For each factual claim in the answer, determine whether it is directly supported by the sources.
Respond as JSON only (no markdown fences): {{"score": 0.0-1.0, "claims": [{{"claim": "...", "supported": true/false, "evidence": "...", "confidence": 0.0-1.0}}], "summary": "..."}}"""

        response = await self.client.messages.create(
            model=self.MODEL,
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
        )

        raw = response.content[0].text
        try:
            data = json.loads(raw)
            return GroundingResult(**data)
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning("Failed to parse grounding response: %s", e)
            return GroundingResult(
                score=0.0,
                claims=[],
                summary=f"Failed to parse grounding evaluation: {e}",
            )
