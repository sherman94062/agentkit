"""Comparable sales analysis and listing price estimator."""

import json
import statistics
from dataclasses import dataclass, field
from datetime import datetime

from demo.db import client as db


@dataclass
class CompAnalysis:
    """Results of a comparable sales analysis."""
    subject_address: str
    subject_acreage: float
    subject_purchase_price: float
    subject_purchase_date: str
    subject_market_value: float

    comps: list[dict] = field(default_factory=list)
    comp_count: int = 0

    # Price per acre stats from comps
    ppa_low: float = 0.0
    ppa_median: float = 0.0
    ppa_high: float = 0.0
    ppa_mean: float = 0.0

    # Estimated value range
    estimated_low: float = 0.0
    estimated_mid: float = 0.0
    estimated_high: float = 0.0

    # Appreciation
    appreciation_pct: float = 0.0
    annual_appreciation_pct: float = 0.0
    years_held: float = 0.0

    # Market context
    market_notes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"Comparable Sales Analysis: {self.subject_address}",
            f"{'=' * 60}",
            f"Subject: {self.subject_acreage} acres, purchased ${self.subject_purchase_price:,.0f} ({self.subject_purchase_date})",
            f"Current market value (CAD): ${self.subject_market_value:,.0f}",
            f"Years held: {self.years_held:.1f}",
            f"",
            f"Comparables found: {self.comp_count}",
        ]
        if self.comp_count > 0:
            lines.extend([
                f"Price/acre range: ${self.ppa_low:,.0f} - ${self.ppa_high:,.0f}",
                f"Price/acre median: ${self.ppa_median:,.0f}",
                f"Price/acre mean:   ${self.ppa_mean:,.0f}",
                f"",
                f"Estimated Value Range ({self.subject_acreage} acres):",
                f"  Low:  ${self.estimated_low:,.0f}  (${self.ppa_low:,.0f}/acre)",
                f"  Mid:  ${self.estimated_mid:,.0f}  (${self.ppa_median:,.0f}/acre)",
                f"  High: ${self.estimated_high:,.0f}  (${self.ppa_high:,.0f}/acre)",
                f"",
                f"Appreciation since purchase: {self.appreciation_pct:.0f}% total, {self.annual_appreciation_pct:.1f}%/year",
            ])
        if self.market_notes:
            lines.append("")
            lines.append("Market Notes:")
            for note in self.market_notes:
                lines.append(f"  - {note}")
        return "\n".join(lines)


@dataclass
class ListingRecommendation:
    """Recommended listing price and strategy."""
    address: str
    recommended_price: float
    price_range_low: float
    price_range_high: float
    price_per_acre: float
    confidence: str  # 'high', 'medium', 'low'

    comp_analysis: CompAnalysis | None = None
    strategy_notes: list[str] = field(default_factory=list)
    pricing_factors: list[dict] = field(default_factory=list)

    # Net proceeds estimate
    gross_proceeds: float = 0.0
    realtor_commission: float = 0.0
    closing_costs: float = 0.0
    estimated_capital_gains_tax: float = 0.0
    net_proceeds: float = 0.0

    def summary(self) -> str:
        lines = [
            f"Listing Price Recommendation: {self.address}",
            f"{'=' * 60}",
            f"",
            f"Recommended listing price: ${self.recommended_price:,.0f}",
            f"Price range: ${self.price_range_low:,.0f} - ${self.price_range_high:,.0f}",
            f"Price per acre: ${self.price_per_acre:,.0f}",
            f"Confidence: {self.confidence}",
        ]
        if self.pricing_factors:
            lines.append("")
            lines.append("Pricing Factors:")
            for f in self.pricing_factors:
                direction = "+" if f.get("adjustment", 0) >= 0 else ""
                lines.append(f"  {f['factor']}: {direction}{f['adjustment']}% — {f['reason']}")
        if self.strategy_notes:
            lines.append("")
            lines.append("Strategy:")
            for note in self.strategy_notes:
                lines.append(f"  - {note}")
        if self.net_proceeds > 0:
            lines.extend([
                "",
                "Estimated Net Proceeds:",
                f"  Gross:              ${self.gross_proceeds:,.0f}",
                f"  Realtor (6%):      -${self.realtor_commission:,.0f}",
                f"  Closing (~2%):     -${self.closing_costs:,.0f}",
                f"  Cap gains (est):   -${self.estimated_capital_gains_tax:,.0f}",
                f"  Net proceeds:       ${self.net_proceeds:,.0f}",
            ])
        return "\n".join(lines)


def analyze_comps(
    property_id: int,
    search_subdivision: str = "",
    search_county: str = "",
    search_zipcode: str = "",
    min_acreage: float = 0,
    max_acreage: float = 0,
) -> CompAnalysis:
    """Run a comparable sales analysis for a property."""
    # Get subject property
    props = db.get_properties()
    subject = next((p for p in props if p["id"] == property_id), None)
    if not subject:
        raise ValueError(f"Property {property_id} not found")

    meta = json.loads(subject.get("metadata_json", "{}"))
    acreage = subject.get("lot_sqft", 0) / 43560 if subject.get("lot_sqft") else 0
    purchase_price = meta.get("purchase_price", 0)
    purchase_date = meta.get("purchase_date", meta.get("deed_date", ""))
    market_value = meta.get("market_value_2025", meta.get("assessed_value_2025", 0))

    # Determine search params
    if not search_subdivision:
        search_subdivision = meta.get("subdivision", "")
    if not search_county:
        search_county = meta.get("county", "")
    if not search_zipcode:
        search_zipcode = subject.get("zipcode", "")

    # Default acreage range: +/- 50% of subject
    if min_acreage == 0 and acreage > 0:
        min_acreage = acreage * 0.5
    if max_acreage == 0 and acreage > 0:
        max_acreage = acreage * 1.5

    # Fetch comps
    comps = db.get_comparables(
        subdivision=search_subdivision,
        county=search_county,
        zipcode=search_zipcode,
        min_acreage=min_acreage,
        max_acreage=max_acreage,
    )

    # Also search broader (county-level) if subdivision has few results
    if len(comps) < 3 and search_county:
        broader = db.get_comparables(
            county=search_county,
            min_acreage=min_acreage,
            max_acreage=max_acreage,
        )
        existing_ids = {c["id"] for c in comps}
        for c in broader:
            if c["id"] not in existing_ids:
                comps.append(c)

    # Calculate stats
    analysis = CompAnalysis(
        subject_address=subject["address"],
        subject_acreage=acreage,
        subject_purchase_price=purchase_price,
        subject_purchase_date=purchase_date,
        subject_market_value=market_value,
        comps=comps,
        comp_count=len(comps),
    )

    if comps:
        ppas = [c["price_per_acre"] for c in comps if c.get("price_per_acre", 0) > 0]
        sale_prices = [c["sale_price"] for c in comps if c.get("sale_price", 0) > 0]

        if ppas:
            analysis.ppa_low = min(ppas)
            analysis.ppa_high = max(ppas)
            analysis.ppa_median = statistics.median(ppas)
            analysis.ppa_mean = statistics.mean(ppas)

            analysis.estimated_low = analysis.ppa_low * acreage
            analysis.estimated_mid = analysis.ppa_median * acreage
            analysis.estimated_high = analysis.ppa_high * acreage

    # Appreciation calc
    if purchase_price > 0 and purchase_date:
        try:
            # Parse various date formats
            for fmt in ["%m/%d/%Y", "%Y-%m-%d", "%m/%d/%y"]:
                try:
                    pd = datetime.strptime(purchase_date, fmt)
                    break
                except ValueError:
                    continue
            else:
                pd = datetime(2016, 1, 1)  # fallback

            years = (datetime.now() - pd).days / 365.25
            analysis.years_held = years

            # Use mid estimate or market value for appreciation
            current = analysis.estimated_mid if analysis.estimated_mid > 0 else market_value
            if current > 0:
                analysis.appreciation_pct = ((current - purchase_price) / purchase_price) * 100
                if years > 0:
                    analysis.annual_appreciation_pct = (
                        ((current / purchase_price) ** (1 / years)) - 1
                    ) * 100
        except Exception:
            pass

    # Market notes
    if market_value > 0 and purchase_price > 0:
        cad_appreciation = ((market_value - purchase_price) / purchase_price) * 100
        analysis.market_notes.append(
            f"CAD market value ${market_value:,.0f} represents {cad_appreciation:.0f}% appreciation over purchase price"
        )

    value_history = meta.get("value_history", [])
    if len(value_history) >= 2:
        latest = value_history[0]
        oldest = value_history[-1]
        if latest.get("land_market", 0) > 0 and oldest.get("land_market", 0) > 0:
            cad_growth = ((latest["land_market"] - oldest["land_market"]) / oldest["land_market"]) * 100
            analysis.market_notes.append(
                f"CAD land value grew {cad_growth:.0f}% from {oldest['year']} to {latest['year']}"
            )
        # Note the 2021-2022 jump if present
        for i, vh in enumerate(value_history):
            if vh["year"] == 2022 and i + 1 < len(value_history):
                prev = value_history[i + 1]
                if prev["year"] == 2021 and prev.get("land_market", 0) > 0:
                    jump = ((vh["land_market"] - prev["land_market"]) / prev["land_market"]) * 100
                    if jump > 50:
                        analysis.market_notes.append(
                            f"Major revaluation in 2022: +{jump:.0f}% (${prev['land_market']:,} → ${vh['land_market']:,})"
                        )

    community = meta.get("community", {})
    if community:
        total = community.get("total_lots", 0)
        with_homes = community.get("lots_with_homes", "")
        if total:
            analysis.market_notes.append(
                f"Community: {total} lots, {with_homes} with homes — active building indicates demand"
            )
        if not community.get("water"):
            analysis.market_notes.append("No municipal water/sewer — well and septic required (adds $30-60k to build cost)")

    return analysis


def estimate_listing_price(
    property_id: int,
    comp_analysis: CompAnalysis | None = None,
) -> ListingRecommendation:
    """Generate a listing price recommendation based on comps and market data."""
    if comp_analysis is None:
        comp_analysis = analyze_comps(property_id)

    props = db.get_properties()
    subject = next((p for p in props if p["id"] == property_id), None)
    meta = json.loads(subject.get("metadata_json", "{}"))
    acreage = comp_analysis.subject_acreage
    purchase_price = comp_analysis.subject_purchase_price

    # Start with the comp-based median estimate
    if comp_analysis.ppa_median > 0:
        base_price = comp_analysis.estimated_mid
        confidence = "medium" if comp_analysis.comp_count >= 3 else "low"
    elif comp_analysis.subject_market_value > 0:
        # Fall back to CAD market value
        base_price = comp_analysis.subject_market_value
        confidence = "low"
    else:
        base_price = purchase_price * 2.5  # rough estimate
        confidence = "low"

    # Apply adjustments
    factors = []

    # Ag exemption — makes the lot cheaper to hold, attractive to buyers
    if meta.get("ag_use_value"):
        factors.append({
            "factor": "Ag exemption in place",
            "adjustment": 5,
            "reason": "Low holding costs (~$850/yr tax) transferable to buyer if maintained",
        })

    # Community development (half built = demand signal)
    community = meta.get("community", {})
    if community.get("lots_with_homes"):
        factors.append({
            "factor": "Active community development",
            "adjustment": 5,
            "reason": "~50% of lots have homes, proving buildability and buyer interest",
        })

    # Infrastructure limitations
    if community and not community.get("water"):
        factors.append({
            "factor": "No municipal water/sewer",
            "adjustment": -10,
            "reason": "Buyer must drill well + install septic ($30-60k additional cost)",
        })

    if community.get("paved_roads") and community.get("electricity"):
        factors.append({
            "factor": "Paved roads + electricity",
            "adjustment": 5,
            "reason": "Key infrastructure in place, reduces build cost vs raw land",
        })

    # Location / market trend
    value_history = meta.get("value_history", [])
    if len(value_history) >= 2:
        recent_years = [v for v in value_history if v["year"] >= 2022]
        if len(recent_years) >= 2:
            latest = recent_years[0]["land_market"]
            earlier = recent_years[-1]["land_market"]
            if latest > earlier:
                factors.append({
                    "factor": "Rising CAD values (2022-2025)",
                    "adjustment": 3,
                    "reason": f"Land market value stable/rising: ${earlier:,} → ${latest:,}",
                })
            elif latest < earlier:
                factors.append({
                    "factor": "Declining CAD values",
                    "adjustment": -5,
                    "reason": f"Land market value declining: ${earlier:,} → ${latest:,}",
                })

    # Multiple lots discount/premium
    # (Seller has 3 lots — could offer bulk deal)
    factors.append({
        "factor": "Multi-lot portfolio",
        "adjustment": 0,
        "reason": "3 adjacent lots available — can offer bundle discount or sell individually",
    })

    # Calculate adjusted price
    total_adjustment = sum(f["adjustment"] for f in factors)
    adjusted_price = base_price * (1 + total_adjustment / 100)

    # Round to nearest $5k
    recommended = round(adjusted_price / 5000) * 5000
    price_low = round(recommended * 0.9 / 5000) * 5000
    price_high = round(recommended * 1.1 / 5000) * 5000

    # Net proceeds estimate
    gross = recommended
    commission = gross * 0.06
    closing = gross * 0.02
    # Long-term capital gains (held > 1 year): 15% federal on gain
    gain = max(0, gross - purchase_price)
    cap_gains = gain * 0.15  # simplified — actual rate depends on income
    net = gross - commission - closing - cap_gains

    rec = ListingRecommendation(
        address=comp_analysis.subject_address,
        recommended_price=recommended,
        price_range_low=price_low,
        price_range_high=price_high,
        price_per_acre=round(recommended / acreage) if acreage > 0 else 0,
        confidence=confidence,
        comp_analysis=comp_analysis,
        strategy_notes=[],
        pricing_factors=factors,
        gross_proceeds=gross,
        realtor_commission=commission,
        closing_costs=closing,
        estimated_capital_gains_tax=cap_gains,
        net_proceeds=net,
    )

    # Strategy notes
    rec.strategy_notes.append(
        f"List at ${recommended:,.0f}/lot (${rec.price_per_acre:,.0f}/acre) — "
        f"competitive for 5-acre Hill Country lots with infrastructure"
    )
    rec.strategy_notes.append(
        "Consider listing on LandWatch, Zillow, and Lands of Texas — these are the top rural land platforms"
    )
    rec.strategy_notes.append(
        "Highlight: paved roads, electricity, ag exemption, established community with homes"
    )
    if purchase_price > 0:
        roi = ((recommended - purchase_price) / purchase_price) * 100
        rec.strategy_notes.append(
            f"ROI since purchase: {roi:.0f}% (${purchase_price:,.0f} → ${recommended:,.0f})"
        )
    rec.strategy_notes.append(
        "For 3-lot bundle: consider 5-10% discount ($X total) to attract investors or builders"
    )

    return rec
