"""Pydantic settings — loads from .env file."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # LLM
    anthropic_api_key: str = ""

    # Data APIs
    rentcast_api_key: str = ""
    attom_api_key: str = ""
    walkscore_api_key: str = ""

    # Database
    db_url: str = "sqlite:///./agentkit_demo.db"

    # Server
    log_level: str = "INFO"
    host: str = "0.0.0.0"
    port: int = 8000

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
