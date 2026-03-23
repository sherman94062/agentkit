"""US Census ACS client — demographics, income, population (free, no key required)."""

import httpx

BASE_URL = "https://api.census.gov/data"
# Using ACS 5-Year Estimates — most reliable for small geographies
ACS_YEAR = "2022"
ACS_DATASET = "acs/acs5"


async def get_demographics(zipcode: str) -> dict:
    """Get key demographic data for a ZIP code from ACS 5-Year estimates."""
    # Variables: total pop, median income, median age, owner-occupied, renter-occupied
    variables = [
        "B01003_001E",  # Total population
        "B19013_001E",  # Median household income
        "B01002_001E",  # Median age
        "B25003_002E",  # Owner-occupied units
        "B25003_003E",  # Renter-occupied units
        "B25077_001E",  # Median home value
        "B25064_001E",  # Median gross rent
    ]

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(
            f"{BASE_URL}/{ACS_YEAR}/{ACS_DATASET}",
            params={
                "get": ",".join(variables),
                "for": f"zip code tabulation area:{zipcode}",
            },
        )
        resp.raise_for_status()
        data = resp.json()

    if len(data) < 2:
        return {"error": f"No census data found for ZIP {zipcode}"}

    headers = data[0]
    values = data[1]
    raw = dict(zip(headers, values))

    def _int(key: str) -> int | None:
        v = raw.get(key)
        if v and v != "-666666666":
            return int(v)
        return None

    return {
        "zipcode": zipcode,
        "total_population": _int("B01003_001E"),
        "median_household_income": _int("B19013_001E"),
        "median_age": _int("B01002_001E"),
        "owner_occupied_units": _int("B25003_002E"),
        "renter_occupied_units": _int("B25003_003E"),
        "median_home_value": _int("B25077_001E"),
        "median_gross_rent": _int("B25064_001E"),
        "source": f"US Census ACS 5-Year Estimates ({ACS_YEAR})",
    }
