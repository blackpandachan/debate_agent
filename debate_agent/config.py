"""Runtime configuration for the debate workflow, read from the environment.

Environment variables (all optional):

  DEBATE_MODEL       Model for the stance and debater agents.
  JUDGE_MODEL        Model for the moderator and final judge (defaults to
                     DEBATE_MODEL). ``claude-haiku-5-5`` makes Claude the judge.
  FACT_CHECK_MODEL   Model for the per-round fact-checker. It needs Google
                     Search, so it must be Gemini; defaults to JUDGE_MODEL if
                     that is Gemini, else DEBATE_MODEL.
  MAX_DEBATE_ROUNDS  Default number of rounds when the session state does not
                     set ``max_rounds`` (default 3).
  ENABLE_SEARCH      "true"/"false": Google Search grounding for debaters
                     (default true; Gemini debaters only).
  ENABLE_FACT_CHECK  "true"/"false": run the fact-checker each round before
                     the moderator scores it (default true; needs search).
  LLM_RETRY_ATTEMPTS Retry attempts for transient Gemini API errors (default 5).

Model routing:

  ``claude-*``    ADK's native ``AnthropicLlm`` (Anthropic API; needs
                  ``ANTHROPIC_API_KEY``).
  ``provider/x``  ADK's ``LiteLlm`` wrapper (needs ``pip install litellm``),
                  e.g. ``openai/gpt-5``.
  anything else   Gemini.
"""

from __future__ import annotations

import os

from google.adk.models.base_llm import BaseLlm
from google.adk.models.google_llm import Gemini
from google.genai import types

DEFAULT_MODEL = "gemini-3.5-flash"
DEFAULT_MAX_ROUNDS = 3


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def max_rounds_default() -> int:
    return max(1, int(os.getenv("MAX_DEBATE_ROUNDS", DEFAULT_MAX_ROUNDS)))


def debate_model_name() -> str:
    return os.getenv("DEBATE_MODEL") or DEFAULT_MODEL


def judge_model_name() -> str:
    return os.getenv("JUDGE_MODEL") or debate_model_name()


def is_claude(model_name: str) -> bool:
    return model_name.startswith("claude-")


def is_gemini(model_name: str) -> bool:
    return "/" not in model_name and not is_claude(model_name)


def search_enabled(model_name: str) -> bool:
    """Google Search grounding is a Gemini built-in tool."""
    return is_gemini(model_name) and _env_bool("ENABLE_SEARCH", True)


def fact_check_model_name() -> str | None:
    """The Gemini model for the fact-checker, or None if fact-checking is off."""
    if not (_env_bool("ENABLE_SEARCH", True) and _env_bool("ENABLE_FACT_CHECK", True)):
        return None
    explicit = os.getenv("FACT_CHECK_MODEL")
    if explicit:
        return explicit if is_gemini(explicit) else None
    for name in (judge_model_name(), debate_model_name()):
        if is_gemini(name):
            return name
    return None


def build_model(model_name: str) -> BaseLlm:
    """Returns an ADK model object for ``model_name``."""
    if is_claude(model_name):
        # Imported lazily: the anthropic package is only needed for Claude.
        from google.adk.models.anthropic_llm import AnthropicLlm

        return AnthropicLlm(model=model_name)
    if is_gemini(model_name):
        return Gemini(
            model=model_name,
            retry_options=types.HttpRetryOptions(
                attempts=int(os.getenv("LLM_RETRY_ATTEMPTS", "5")),
            ),
        )
    from google.adk.models.lite_llm import LiteLlm

    return LiteLlm(model=model_name)


def generate_config(model_name: str) -> types.GenerateContentConfig | None:
    """Per-model generation settings for an agent (None = model defaults).

    Claude agents get adaptive thinking, set explicitly: current Claude models
    think by default, and telling ADK so makes it offer the output-schema tool
    rather than force it (forced tool use is rejected while thinking is on).
    """
    if not is_claude(model_name):
        return None
    return types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_budget=-1)
    )
