"""Centralised configuration loaded from environment variables."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings.

    Values are read from environment variables and/or a `.env` file in the
    project root. See `.env.example` for the full list with documentation.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Credentials
    anthropic_api_key: str = Field(..., description="Anthropic API key")
    openai_api_key: str | None = Field(default=None)

    # Observability
    langsmith_api_key: str | None = Field(default=None)
    langsmith_project: str = Field(default="contact-center-copilot")
    langsmith_tracing: bool = Field(default=False)

    # Models
    model_fast: str = Field(default="claude-haiku-4-5-20251001")
    model_heavy: str = Field(default="claude-sonnet-4-6")
    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")

    # Pipeline behaviour
    faithfulness_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    max_reporter_attempts: int = Field(default=3, ge=1, le=10)
    retriever_top_k: int = Field(default=5, ge=1, le=50)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings instance (read once per process)."""
    return Settings()
