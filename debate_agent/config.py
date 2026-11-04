"""Runtime configuration for the debate workflow, read from the environment.

Environment variables (all optional):

  DEBATE_MODEL       Model for the stance and debater agents.
  JUDGE_MODEL        Model for the moderator and final judge (defaults to
                     DEBATE_MODEL).
  MAX_DEBATE_ROUNDS  Default number of rounds when the session state does not
                     set ``max_rounds`` (default 3).
  ENABLE_SEARCH      "true"/"false": give debaters Google Search grounding
                     (default true; only applies to Gemini models).
  ENABLE_FACT_CHECK  "true"/"false": run a search-backed fact-checker each
                     round before the moderator scores it (default true;
                     needs search, so Gemini only).
  LLM_RETRY_ATTEMPTS Retry attempts for transient Gemini API errors (default 5).

Model strings containing a "/" (for example ``openai/gpt-5`` or
``anthropic/claude-sonnet-5-5``) are routed through ADK's ``LiteLlm`` wrapper and
need ``pip install litellm`` plus that provider's API key. Anything else is
treated as a Gemini model id.
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


def is_gemini(model_name: str) -> bool:
    return "/" not in model_name


def search_enabled(model_name: str) -> bool:
    """Google Search grounding is a Gemini built-in tool."""
    return is_gemini(model_name) and _env_bool("ENABLE_SEARCH", True)


def fact_check_enabled(model_name: str) -> bool:
    return search_enabled(model_name) and _env_bool("ENABLE_FACT_CHECK", True)


def build_model(model_name: str) -> BaseLlm:
    """Returns an ADK model object for ``model_name``."""
    if is_gemini(model_name):
        return Gemini(
            model=model_name,
            retry_options=types.HttpRetryOptions(
                attempts=int(os.getenv("LLM_RETRY_ATTEMPTS", "5")),
            ),
        )
    # Imported lazily: LiteLLM is an optional dependency.
    from google.adk.models.lite_llm import LiteLlm

    return LiteLlm(model=model_name)
