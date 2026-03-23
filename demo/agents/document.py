"""DocumentAgent — parse uploaded PDFs (deed, inspection, HOA, appraisal)."""

import json

import anthropic

from agentkit import AgentContext, AgentResult, BaseAgent, InteractionLogger, ToolRegistry

from demo.agents.market import _extract_final_text, _extract_tool_results
from demo.db import client as db


class DocumentAgent(BaseAgent):

    @property
    def name(self) -> str:
        return "DocumentAgent"

    @property
    def system_prompt(self) -> str:
        return (
            "You are a real estate document analyst. Extract factual information from "
            "property documents. For inspection reports, categorize issues as "
            "major/minor/cosmetic and flag anything that affects safety or habitability. "
            "Always cite page numbers when available."
        )

    @property
    def tool_names(self) -> list[str]:
        return ["list_documents", "extract_deed_data", "extract_inspection_issues"]

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
            confidence=0.85 if sources else 0.2,
            tokens_used=usage[0] + usage[1],
            cost_usd=self.interaction_logger.total_cost,
        )


def _list_documents(address: str) -> str:
    """List uploaded documents for a property address."""
    docs = db.get_documents(address)
    if not docs:
        return json.dumps({"documents": [], "message": f"No documents found for {address}"})
    return json.dumps({
        "documents": [
            {
                "id": d["id"],
                "filename": d["filename"],
                "doc_type": d["doc_type"],
                "uploaded_at": d["uploaded_at"],
            }
            for d in docs
        ]
    })


def _extract_deed_data(doc_id: int) -> str:
    """Extract structured data from a deed document."""
    docs = db.get_documents()
    doc = next((d for d in docs if d["id"] == doc_id), None)
    if not doc:
        return json.dumps({"error": f"Document {doc_id} not found"})

    extracted = json.loads(doc.get("extracted_json", "{}"))
    return json.dumps({
        "doc_id": doc_id,
        "filename": doc["filename"],
        "doc_type": doc["doc_type"],
        "content_preview": (doc.get("content_text") or "")[:1000],
        "extracted_fields": extracted,
    })


def _extract_inspection_issues(doc_id: int) -> str:
    """Extract inspection issues categorized by severity."""
    docs = db.get_documents()
    doc = next((d for d in docs if d["id"] == doc_id), None)
    if not doc:
        return json.dumps({"error": f"Document {doc_id} not found"})

    extracted = json.loads(doc.get("extracted_json", "{}"))
    return json.dumps({
        "doc_id": doc_id,
        "filename": doc["filename"],
        "content_preview": (doc.get("content_text") or "")[:2000],
        "extracted_issues": extracted.get("issues", []),
    })


def register_tools(registry: ToolRegistry) -> None:
    """Register DocumentAgent tools."""
    registry.add(
        "list_documents",
        _list_documents,
        "List uploaded documents for a property address.",
        tags=["read", "local"],
        input_schema={
            "type": "object",
            "properties": {
                "address": {"type": "string", "description": "Property address"},
            },
            "required": ["address"],
        },
    )

    registry.add(
        "extract_deed_data",
        _extract_deed_data,
        "Extract structured data from a deed document by doc ID.",
        tags=["read", "local"],
        input_schema={
            "type": "object",
            "properties": {
                "doc_id": {"type": "integer", "description": "Document ID from list_documents"},
            },
            "required": ["doc_id"],
        },
    )

    registry.add(
        "extract_inspection_issues",
        _extract_inspection_issues,
        "Extract inspection report issues categorized by severity (major/minor/cosmetic).",
        tags=["read", "local"],
        input_schema={
            "type": "object",
            "properties": {
                "doc_id": {"type": "integer", "description": "Document ID from list_documents"},
            },
            "required": ["doc_id"],
        },
    )
