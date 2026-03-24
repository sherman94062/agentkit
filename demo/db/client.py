"""Async SQLite client for the demo app."""

import json
import sqlite3
from pathlib import Path
from typing import Any

DB_PATH = Path("agentkit_demo.db")
SCHEMA_PATH = Path(__file__).parent / "schema.sql"


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db() -> None:
    """Create tables if they don't exist."""
    conn = _get_conn()
    conn.executescript(SCHEMA_PATH.read_text())
    conn.close()


def save_property(
    address: str,
    city: str = "",
    state: str = "",
    zipcode: str = "",
    lat: float = 0.0,
    lon: float = 0.0,
    bedrooms: int = 0,
    bathrooms: float = 0.0,
    sqft: int = 0,
    lot_sqft: int = 0,
    year_built: int = 0,
    property_type: str = "",
    metadata: dict | None = None,
) -> int:
    conn = _get_conn()
    cur = conn.execute(
        """INSERT INTO properties
           (address, city, state, zipcode, lat, lon, bedrooms, bathrooms,
            sqft, lot_sqft, year_built, property_type, metadata_json)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            address, city, state, zipcode, lat, lon,
            bedrooms, bathrooms, sqft, lot_sqft, year_built,
            property_type, json.dumps(metadata or {}),
        ),
    )
    conn.commit()
    row_id = cur.lastrowid
    conn.close()
    return row_id


def get_properties() -> list[dict]:
    conn = _get_conn()
    rows = conn.execute("SELECT * FROM properties ORDER BY created_at DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def save_document(
    address: str,
    filename: str,
    doc_type: str,
    content_text: str,
    extracted_json: dict | None = None,
    property_id: int | None = None,
) -> int:
    conn = _get_conn()
    cur = conn.execute(
        """INSERT INTO documents (property_id, address, filename, doc_type, content_text, extracted_json)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (property_id, address, filename, doc_type, content_text, json.dumps(extracted_json or {})),
    )
    conn.commit()
    row_id = cur.lastrowid
    conn.close()
    return row_id


def get_documents(address: str = "") -> list[dict]:
    conn = _get_conn()
    if address:
        rows = conn.execute(
            "SELECT * FROM documents WHERE address = ? ORDER BY uploaded_at DESC", (address,)
        ).fetchall()
    else:
        rows = conn.execute("SELECT * FROM documents ORDER BY uploaded_at DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def save_query(
    session_id: str,
    user_query: str,
    answer: str,
    grounding_score: float = 0.0,
    agent_trace: list | None = None,
    total_cost: float = 0.0,
    latency_s: float = 0.0,
) -> int:
    conn = _get_conn()
    cur = conn.execute(
        """INSERT INTO queries (session_id, user_query, answer, grounding_score,
           agent_trace_json, total_cost, latency_s)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (session_id, user_query, answer, grounding_score,
         json.dumps(agent_trace or []), total_cost, latency_s),
    )
    conn.commit()
    row_id = cur.lastrowid
    conn.close()
    return row_id


def save_comparable(
    address: str = "",
    city: str = "",
    state: str = "",
    zipcode: str = "",
    county: str = "",
    subdivision: str = "",
    lot_number: str = "",
    acreage: float = 0.0,
    property_type: str = "Vacant Land",
    sale_date: str = "",
    sale_price: float = 0.0,
    price_per_acre: float = 0.0,
    market_value: float = 0.0,
    appraised_value: float = 0.0,
    has_improvements: bool = False,
    improvement_value: float = 0.0,
    data_source: str = "manual",
    notes: str = "",
    metadata: dict | None = None,
) -> int:
    conn = _get_conn()
    ppa = price_per_acre or (sale_price / acreage if acreage > 0 and sale_price > 0 else 0)
    cur = conn.execute(
        """INSERT INTO comparables
           (address, city, state, zipcode, county, subdivision, lot_number,
            acreage, property_type, sale_date, sale_price, price_per_acre,
            market_value, appraised_value, has_improvements, improvement_value,
            data_source, notes, metadata_json)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            address, city, state, zipcode, county, subdivision, lot_number,
            acreage, property_type, sale_date, sale_price, ppa,
            market_value, appraised_value, int(has_improvements), improvement_value,
            data_source, notes, json.dumps(metadata or {}),
        ),
    )
    conn.commit()
    row_id = cur.lastrowid
    conn.close()
    return row_id


def get_comparables(
    subdivision: str = "",
    county: str = "",
    zipcode: str = "",
    min_acreage: float = 0,
    max_acreage: float = 0,
) -> list[dict]:
    conn = _get_conn()
    query = "SELECT * FROM comparables WHERE 1=1"
    params = []
    if subdivision:
        query += " AND subdivision = ?"
        params.append(subdivision)
    if county:
        query += " AND county = ?"
        params.append(county)
    if zipcode:
        query += " AND zipcode = ?"
        params.append(zipcode)
    if min_acreage > 0:
        query += " AND acreage >= ?"
        params.append(min_acreage)
    if max_acreage > 0:
        query += " AND acreage <= ?"
        params.append(max_acreage)
    query += " ORDER BY sale_date DESC"
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]
