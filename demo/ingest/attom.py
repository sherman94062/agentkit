"""ATTOM API client — parcel details, tax history, sale history, AVM."""

import httpx

from demo.config import settings

BASE_URL = "https://api.gateway.attomdata.com/propertyapi/v1.0.0"


def _headers() -> dict:
    return {"apikey": settings.attom_api_key, "Accept": "application/json"}


def _split_address(address: str) -> dict:
    """Split 'street, city, ST zip' into ATTOM's address1 + address2 params."""
    parts = [p.strip() for p in address.split(",")]
    if len(parts) >= 2:
        return {"address1": parts[0], "address2": ", ".join(parts[1:])}
    return {"address1": address, "address2": ""}


async def get_parcel(address: str) -> dict:
    """Get basic property/parcel profile by address."""
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(
            f"{BASE_URL}/property/basicprofile",
            headers=_headers(),
            params=_split_address(address),
        )
        resp.raise_for_status()
        return resp.json()


async def get_tax_history(attom_id: str) -> dict:
    """Get tax assessment history for a property."""
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(
            f"{BASE_URL}/assessment/history",
            headers=_headers(),
            params={"attomId": attom_id},
        )
        resp.raise_for_status()
        return resp.json()


async def get_sale_history(attom_id: str) -> dict:
    """Get sale/transfer history for a property."""
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(
            f"{BASE_URL}/saleshistory/detail",
            headers=_headers(),
            params={"attomId": attom_id},
        )
        resp.raise_for_status()
        return resp.json()


async def get_avm(address: str) -> dict:
    """Get automated valuation model estimate."""
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(
            f"{BASE_URL}/avm/detail",
            headers=_headers(),
            params=_split_address(address),
        )
        resp.raise_for_status()
        return resp.json()
