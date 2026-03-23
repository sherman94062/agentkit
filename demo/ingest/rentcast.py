"""Rentcast API client — listings, comps, rent estimates, market stats."""

import httpx

from demo.config import settings

BASE_URL = "https://api.rentcast.io/v1"


def _headers() -> dict:
    return {"X-Api-Key": settings.rentcast_api_key, "Accept": "application/json"}


async def search_listings(
    address: str, radius_miles: float = 5.0, max_results: int = 10
) -> dict:
    """Search active sale listings near an address."""
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(
            f"{BASE_URL}/listings/sale",
            headers=_headers(),
            params={
                "address": address,
                "radius": radius_miles,
                "limit": max_results,
                "status": "Active",
            },
        )
        resp.raise_for_status()
        return resp.json()


async def get_comps(
    address: str, radius_miles: float = 5.0, sold_in_days: int = 180
) -> dict:
    """Get comparable sales for an address."""
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(
            f"{BASE_URL}/avm/sale/comparables",
            headers=_headers(),
            params={
                "address": address,
                "radius": radius_miles,
                "daysOld": sold_in_days,
            },
        )
        resp.raise_for_status()
        return resp.json()


async def get_market_stats(zipcode: str) -> dict:
    """Get market statistics for a ZIP code."""
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(
            f"{BASE_URL}/markets",
            headers=_headers(),
            params={"zipCode": zipcode},
        )
        resp.raise_for_status()
        return resp.json()


async def estimate_rent(
    address: str, bedrooms: int = 3, bathrooms: float = 2.0
) -> dict:
    """Get long-term rental estimate for an address."""
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(
            f"{BASE_URL}/avm/rent/long-term",
            headers=_headers(),
            params={
                "address": address,
                "bedrooms": bedrooms,
                "bathrooms": bathrooms,
            },
        )
        resp.raise_for_status()
        return resp.json()
