"""FastAPI application — Real Estate Property Intelligence Demo."""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from demo.config import settings
from demo.db import client as db
from demo.routers import documents, properties, query, ui_api

logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO))
logger = logging.getLogger("agentkit.demo")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize database on startup."""
    logger.info("Initializing database...")
    db.init_db()
    # Wire the shared logger into the UI API
    ui_api.set_logger(query._logger)
    logger.info("agentkit demo ready — %d tools registered", len(query._registry))
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="agentkit Real Estate Demo",
    description="Multi-agent property intelligence powered by agentkit",
    version="0.1.0",
    lifespan=lifespan,
)

# Routers
app.include_router(query.router)
app.include_router(documents.router)
app.include_router(properties.router)
app.include_router(ui_api.router)

# Static files (dashboard SPA)
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")
