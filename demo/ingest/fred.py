"""FRED API client — mortgage rates and economic data (free, no key required)."""

import httpx

BASE_URL = "https://api.stlouisfed.org/fred"
# FRED is free but does require a free API key for structured endpoints.
# For public series we can use the observations endpoint.
FRED_API_KEY = "DEMO_KEY"  # Replace with free key from https://fred.stlouisfed.org/docs/api/api_key.html


async def get_current_mortgage_rates() -> dict:
    """Get the latest 30-year fixed mortgage rate (MORTGAGE30US series)."""
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(
            f"{BASE_URL}/series/observations",
            params={
                "series_id": "MORTGAGE30US",
                "api_key": FRED_API_KEY,
                "file_type": "json",
                "sort_order": "desc",
                "limit": 4,  # last 4 weeks
            },
        )
        resp.raise_for_status()
        data = resp.json()
        observations = data.get("observations", [])
        if not observations:
            return {"rate": None, "date": None, "error": "No data available"}

        latest = observations[0]
        return {
            "rate": float(latest["value"]),
            "date": latest["date"],
            "series": "MORTGAGE30US",
            "description": "30-Year Fixed Rate Mortgage Average (Freddie Mac)",
            "history": [
                {"date": o["date"], "rate": float(o["value"])}
                for o in observations
                if o["value"] != "."
            ],
        }


async def get_series(series_id: str, limit: int = 12) -> dict:
    """Get recent observations for any FRED series."""
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(
            f"{BASE_URL}/series/observations",
            params={
                "series_id": series_id,
                "api_key": FRED_API_KEY,
                "file_type": "json",
                "sort_order": "desc",
                "limit": limit,
            },
        )
        resp.raise_for_status()
        return resp.json()
