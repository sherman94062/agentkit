"""Valuation router — comp analysis and listing price estimates."""

from fastapi import APIRouter, HTTPException

from demo.db import client as db
from demo.valuation import analyze_comps, estimate_listing_price

router = APIRouter(prefix="/api/valuation", tags=["valuation"])


@router.get("/comps/{property_id}")
async def get_comp_analysis(property_id: int):
    """Run comparable sales analysis for a property."""
    try:
        analysis = analyze_comps(property_id)
        return {
            "subject": {
                "address": analysis.subject_address,
                "acreage": analysis.subject_acreage,
                "purchase_price": analysis.subject_purchase_price,
                "purchase_date": analysis.subject_purchase_date,
                "market_value": analysis.subject_market_value,
                "years_held": analysis.years_held,
                "appreciation_pct": analysis.appreciation_pct,
                "annual_appreciation_pct": analysis.annual_appreciation_pct,
            },
            "comps": [
                {
                    "address": c.get("address", ""),
                    "subdivision": c.get("subdivision", ""),
                    "lot_number": c.get("lot_number", ""),
                    "acreage": c.get("acreage", 0),
                    "sale_date": c.get("sale_date", ""),
                    "sale_price": c.get("sale_price", 0),
                    "price_per_acre": c.get("price_per_acre", 0),
                    "market_value": c.get("market_value", 0),
                    "has_improvements": bool(c.get("has_improvements", 0)),
                    "data_source": c.get("data_source", ""),
                    "notes": c.get("notes", ""),
                }
                for c in analysis.comps
            ],
            "stats": {
                "comp_count": analysis.comp_count,
                "ppa_low": analysis.ppa_low,
                "ppa_median": analysis.ppa_median,
                "ppa_high": analysis.ppa_high,
                "ppa_mean": analysis.ppa_mean,
            },
            "estimated_value": {
                "low": analysis.estimated_low,
                "mid": analysis.estimated_mid,
                "high": analysis.estimated_high,
            },
            "market_notes": analysis.market_notes,
            "summary": analysis.summary(),
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/price/{property_id}")
async def get_listing_price(property_id: int):
    """Get listing price recommendation for a property."""
    try:
        rec = estimate_listing_price(property_id)
        return {
            "address": rec.address,
            "recommended_price": rec.recommended_price,
            "price_range": {
                "low": rec.price_range_low,
                "high": rec.price_range_high,
            },
            "price_per_acre": rec.price_per_acre,
            "confidence": rec.confidence,
            "pricing_factors": rec.pricing_factors,
            "strategy_notes": rec.strategy_notes,
            "net_proceeds": {
                "gross": rec.gross_proceeds,
                "realtor_commission": rec.realtor_commission,
                "closing_costs": rec.closing_costs,
                "capital_gains_tax": rec.estimated_capital_gains_tax,
                "net": rec.net_proceeds,
            },
            "summary": rec.summary(),
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/portfolio")
async def portfolio_summary():
    """Get valuation summary for all Round Mountain properties."""
    props = db.get_properties()
    blanco_props = [p for p in props if p.get("county") == "Blanco" or "Round Mountain" in (p.get("address") or "")]

    # Use property IDs 3, 4, 5 for the Round Mountain lots
    results = []
    total_recommended = 0
    total_net = 0
    total_purchase = 0

    for prop in blanco_props:
        try:
            rec = estimate_listing_price(prop["id"])
            results.append({
                "id": prop["id"],
                "address": prop["address"],
                "recommended_price": rec.recommended_price,
                "price_per_acre": rec.price_per_acre,
                "net_proceeds": rec.net_proceeds,
                "confidence": rec.confidence,
            })
            total_recommended += rec.recommended_price
            total_net += rec.net_proceeds
            total_purchase += rec.comp_analysis.subject_purchase_price if rec.comp_analysis else 0
        except Exception:
            pass

    return {
        "properties": results,
        "portfolio_total": {
            "total_recommended": total_recommended,
            "total_net_proceeds": total_net,
            "total_purchase_price": total_purchase,
            "total_gain": total_net - total_purchase if total_purchase > 0 else 0,
            "total_roi_pct": ((total_net - total_purchase) / total_purchase * 100) if total_purchase > 0 else 0,
        },
    }


@router.post("/comps")
async def add_comparable(data: dict):
    """Add a comparable sale manually."""
    comp_id = db.save_comparable(
        address=data.get("address", ""),
        city=data.get("city", ""),
        state=data.get("state", "TX"),
        zipcode=data.get("zipcode", ""),
        county=data.get("county", ""),
        subdivision=data.get("subdivision", ""),
        lot_number=data.get("lot_number", ""),
        acreage=data.get("acreage", 0),
        property_type=data.get("property_type", "Vacant Land"),
        sale_date=data.get("sale_date", ""),
        sale_price=data.get("sale_price", 0),
        market_value=data.get("market_value", 0),
        appraised_value=data.get("appraised_value", 0),
        has_improvements=data.get("has_improvements", False),
        data_source=data.get("data_source", "manual"),
        notes=data.get("notes", ""),
    )
    return {"id": comp_id}
