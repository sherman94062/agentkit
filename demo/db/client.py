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
