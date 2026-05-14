"""LLM factory — provider-agnostic access to fast/heavy chat models.

Supports Groq (free, fast), Anthropic (paid, higher quality), and Ollama
(free, fully local). Switch via the `LLM_PROVIDER` env variable. Adding more
providers (OpenAI, Gemini, Azure OpenAI) is a small change here — agents
don't need to know which provider is active.
"""

from __future__ import annotations

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

from ..config import get_settings


def _build_groq(model: str, temperature: float) -> BaseChatModel:
    """Lazy import keeps the dependency optional."""
    from langchain_groq import ChatGroq

    settings = get_settings()
    if not settings.groq_api_key:
        raise RuntimeError(
            "GROQ_API_KEY not set. Get one at https://console.groq.com and add it to .env"
        )
    # ChatGroq's pydantic field is `model_name` (`model` is its alias); the
    # mypy plugin doesn't resolve the alias for __init__, hence the ignore.
    return ChatGroq(  # type: ignore[call-arg]
        model=model,
        temperature=temperature,
        api_key=settings.groq_api_key,
    )


def _build_anthropic(model: str, temperature: float) -> BaseChatModel:
    from langchain_anthropic import ChatAnthropic

    settings = get_settings()
    if not settings.anthropic_api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")
    return ChatAnthropic(
        model=model,
        temperature=temperature,
        api_key=settings.anthropic_api_key,
    )


def _build_ollama(model: str, temperature: float) -> BaseChatModel:
    """Lazy import keeps the dependency optional. Ollama is fully local — no key."""
    from langchain_ollama import ChatOllama

    settings = get_settings()
    return ChatOllama(
        model=model,
        temperature=temperature,
        base_url=settings.ollama_base_url,
    )


def _build(model: str, temperature: float) -> BaseChatModel:
    settings = get_settings()
    if settings.llm_provider == "groq":
        return _build_groq(model, temperature)
    if settings.llm_provider == "anthropic":
        return _build_anthropic(model, temperature)
    if settings.llm_provider == "ollama":
        return _build_ollama(model, temperature)
    raise ValueError(f"Unknown LLM provider: {settings.llm_provider}")


def get_fast_llm(temperature: float = 0.0) -> BaseChatModel:
    """Fast/cheap LLM for routing and lightweight tasks."""
    settings = get_settings()
    return _build(settings.model_fast, temperature)


def get_heavy_llm(temperature: float = 0.0) -> BaseChatModel:
    """Heavy LLM for analysis, narrative, critique."""
    settings = get_settings()
    return _build(settings.model_heavy, temperature)


def llm_with_structured_output(model: BaseChatModel, schema: Any) -> Any:
    """Apply structured output binding, handling provider differences.

    Groq supports JSON mode + function calling. Anthropic has native
    Pydantic binding. Both go through LangChain's unified interface.
    """
    return model.with_structured_output(schema)
