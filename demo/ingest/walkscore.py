"""Walk Score API client — walk, transit, and bike scores."""

import httpx

from demo.config import settings

BASE_URL = "https://api.walkscore.com/score"


async def get_walk_score(address: str, lat: float, lon: float) -> dict:
    """Get walk, transit, and bike scores for a location."""
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(
            BASE_URL,
            params={
                "format": "json",
                "address": address,
                "lat": lat,
                "lon": lon,
                "transit": 1,
                "bike": 1,
                "wsapikey": settings.walkscore_api_key,
            },
        )
        resp.raise_for_status()
        return resp.json()
