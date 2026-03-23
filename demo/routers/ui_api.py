"""UI API router — dashboard data: sources, costs, interaction log."""

from fastapi import APIRouter

from demo.config import settings

router = APIRouter(prefix="/api", tags=["ui"])

# Lazy import to avoid circular dependency
_logger = None


def set_logger(logger):
    global _logger
    _logger = logger


@router.get("/sources")
async def data_source_status():
    """Report which data source APIs are configured (have keys)."""
    return {
        "sources": [
            {
                "name": "Rentcast",
                "configured": bool(settings.rentcast_api_key),
                "description": "Listings, comps, rent estimates",
            },
            {
                "name": "ATTOM",
                "configured": bool(settings.attom_api_key),
                "description": "Parcel, tax, AVM",
            },
            {
                "name": "Walk Score",
                "configured": bool(settings.walkscore_api_key),
                "description": "Walk/transit/bike scores",
            },
            {
                "name": "FRED",
                "configured": True,  # Free, always available
                "description": "Mortgage rates, economic data",
            },
            {
                "name": "US Census",
                "configured": True,  # Free, always available
                "description": "Demographics, income, home values",
            },
        ]
    }


@router.get("/session-cost")
async def session_cost():
    """Return token usage and cost for the current session."""
    if _logger is None:
        return {"input_tokens": 0, "output_tokens": 0, "total_cost": 0.0, "calls": 0}

    inp, out = _logger.total_tokens
    return {
        "input_tokens": inp,
        "output_tokens": out,
        "total_cost": _logger.total_cost,
        "calls": len(_logger.entries),
    }


@router.get("/interaction-log")
async def interaction_log():
    """Return the LLM call audit trail."""
    if _logger is None:
        return {"entries": []}
    return {"entries": _logger.to_dicts()}
