"""Properties router — saved property lookups."""

from fastapi import APIRouter

from demo.db import client as db

router = APIRouter(prefix="/properties", tags=["properties"])


@router.get("")
async def list_properties():
    """List all saved properties."""
    props = db.get_properties()
    return {"properties": props}


@router.post("")
async def save_property(data: dict):
    """Save a new property record."""
    prop_id = db.save_property(
        address=data.get("address", ""),
        city=data.get("city", ""),
        state=data.get("state", ""),
        zipcode=data.get("zipcode", ""),
        lat=data.get("lat", 0.0),
        lon=data.get("lon", 0.0),
        bedrooms=data.get("bedrooms", 0),
        bathrooms=data.get("bathrooms", 0.0),
        sqft=data.get("sqft", 0),
        lot_sqft=data.get("lot_sqft", 0),
        year_built=data.get("year_built", 0),
        property_type=data.get("property_type", ""),
        metadata=data.get("metadata"),
    )
    return {"id": prop_id, "address": data.get("address", "")}
