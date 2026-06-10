"""Centralised configuration loaded from environment variables."""

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings — see `.env.example` for documentation."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # LLM provider selection
    llm_provider: Literal["groq", "anthropic", "ollama"] = Field(
        default="ollama",
        description="Active LLM provider. Ollama and Groq are free; Anthropic is paid.",
    )

    # Credentials (set the one matching llm_provider; Ollama needs none)
    groq_api_key: str | None = Field(default=None)
    anthropic_api_key: str | None = Field(default=None)
    openai_api_key: str | None = Field(default=None)

    # Ollama (free, fully local — used when llm_provider == 'ollama')
    ollama_base_url: str = Field(default="http://localhost:11434")

    # Observability
    langsmith_api_key: str | None = Field(default=None)
    langsmith_project: str = Field(default="contact-center-copilot")
    langsmith_tracing: bool = Field(default=False)

    # Models — defaults match the active provider (Ollama). Override per
    # provider via MODEL_FAST / MODEL_HEAVY in .env (see .env.example).
    model_fast: str = Field(default="qwen2.5:14b")
    model_heavy: str = Field(default="qwen2.5:14b")
    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")

    # Pipeline behaviour
    faithfulness_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    max_reporter_attempts: int = Field(default=3, ge=1, le=10)
    retriever_top_k: int = Field(default=5, ge=1, le=50)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached Settings instance (singleton, parsed once from .env)."""
    return Settings()
