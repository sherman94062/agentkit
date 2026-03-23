"""Pydantic request/response models for the API."""

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str
    address: str = ""
    session_id: str = ""


class QueryResponse(BaseModel):
    answer: str
    grounding_score: float = 0.0
    agent_trace: list[dict] = Field(default_factory=list)
    total_cost: float = 0.0
    total_tokens: int = 0
    session_id: str = ""


class DocumentUpload(BaseModel):
    address: str
    doc_type: str = "inspection"  # deed, inspection, hoa, appraisal
    filename: str = ""


class PropertyResponse(BaseModel):
    id: int
    address: str
    city: str = ""
    state: str = ""
    zipcode: str = ""
    bedrooms: int = 0
    bathrooms: float = 0.0
    sqft: int = 0
    year_built: int = 0
    property_type: str = ""
