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


def _retry_exception_types() -> tuple[type[BaseException], ...]:
    """Exception classes that should trigger a Runnable retry.

    Two Groq-specific cases:

    1. ``AuthenticationError`` — Groq's edge fleet sometimes returns
       ``401 Invalid API Key`` for a valid key when the request lands on a
       node that has not yet picked up the key from the auth backend; the
       next request to a different node succeeds.
    2. ``RateLimitError`` — the free tier's TPM ceiling is easy to hit when
       the Reporter fires three structured-output calls in parallel; backoff
       gets us through.

    Without ``groq`` installed there is nothing provider-specific to catch.
    """
    try:
        from groq import AuthenticationError as GroqAuthError
        from groq import RateLimitError as GroqRateLimitError
    except ImportError:
        return ()
    return (GroqAuthError, GroqRateLimitError)


_RETRY_EXC = _retry_exception_types()


def llm_with_structured_output(model: BaseChatModel, schema: Any) -> Any:
    """Apply structured output binding, handling provider differences.

    Groq supports JSON mode + function calling. Anthropic has native
    Pydantic binding. Both go through LangChain's unified interface.

    The returned Runnable retries flaky Groq auth errors (see
    ``_retry_exception_types``); on Ollama/Anthropic the retry is a no-op
    because those exception types are never raised.
    """
    runnable = model.with_structured_output(schema)
    if not _RETRY_EXC:
        return runnable
    return runnable.with_retry(
        retry_if_exception_type=_RETRY_EXC,
        wait_exponential_jitter=True,
        stop_after_attempt=10,
    )
