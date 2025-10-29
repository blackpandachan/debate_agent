from __future__ import annotations

from abc import ABC, abstractmethod
import os
from typing import Any, Dict, Optional

from google.adk.agents import Agent

try:  # The GenerationConfig helper is optional but convenient when available.
    from google.generativeai import types as genai_types  # type: ignore
except ImportError:  # pragma: no cover - the package may not be installed in CI.
    genai_types = None  # type: ignore

from pydantic import PrivateAttr

from ..core.llm_clients import (
    get_anthropic_client,
    get_gemini_client,
    get_openai_client,
)


class BaseLLMAgent(Agent, ABC):
    """Common utilities for agents that rely on LLM completions."""

    llm_service_name: str
    _llm_client: Any = PrivateAttr(default=None)

    def __init__(
        self,
        *,
        agent_id: str,
        llm_service_name: str,
        adk_model_name: Optional[str] = None,
        adk_instruction: Optional[str] = None,
        adk_description: Optional[str] = None,
        adk_tools: Optional[list] = None,
        **additional_agent_kwargs: Any,
    ) -> None:
        agent_kwargs: Dict[str, Any] = {"name": agent_id}
        if adk_model_name:
            agent_kwargs["model"] = adk_model_name
        if adk_instruction:
            agent_kwargs["instruction"] = adk_instruction
        if adk_description:
            agent_kwargs["description"] = adk_description
        if adk_tools:
            agent_kwargs["tools"] = adk_tools
        agent_kwargs.update(additional_agent_kwargs)

        super().__init__(llm_service_name=llm_service_name.lower(), **agent_kwargs)
        self._initialize_llm_client()

    # ------------------------------------------------------------------
    # LLM client bootstrap
    # ------------------------------------------------------------------
    def _initialize_llm_client(self) -> None:
        service = self.llm_service_name
        if service == "gemini":
            gemini_client_or_module = get_gemini_client()
            if not gemini_client_or_module:
                print(
                    f"Agent '{self.name}': Gemini client setup failed; missing credentials or configuration."
                )
                return
            use_vertex_ai = (
                os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "False").lower() == "true"
            )
            if use_vertex_ai:
                self._llm_client = gemini_client_or_module
                print(
                    f"Agent '{self.name}': Initialized Gemini client via Vertex AI ({type(self._llm_client)})."
                )
            else:
                try:
                    model_name = os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-flash-latest")
                    if genai_types is not None:
                        default_config = genai_types.GenerationConfig(candidate_count=1)
                    else:  # pragma: no cover - defensive path when helper is absent
                        default_config = None
                    if default_config is not None:
                        self._llm_client = gemini_client_or_module.GenerativeModel(
                            model_name, generation_config=default_config
                        )
                    else:
                        self._llm_client = gemini_client_or_module.GenerativeModel(model_name)
                    print(
                        f"Agent '{self.name}': Initialized Gemini GenerativeModel '{model_name}'."
                    )
                except Exception as exc:  # pragma: no cover - runtime safety net
                    print(
                        f"Agent '{self.name}': Failed to create Gemini GenerativeModel ({exc})."
                    )
                    self._llm_client = None
        elif service == "openai":
            self._llm_client = get_openai_client()
        elif service == "anthropic":
            self._llm_client = get_anthropic_client()
        else:  # pragma: no cover - validated upstream
            raise ValueError(f"Unsupported LLM service '{service}'.")

    # ------------------------------------------------------------------
    # Public helper
    # ------------------------------------------------------------------
    def invoke_llm(self, prompt: str, **kwargs: Any) -> str:
        """Lightweight wrapper so subclasses need not touch the protected method."""
        return self._invoke_llm(prompt, **kwargs)

    # ------------------------------------------------------------------
    # Concrete LLM invocation plumbing
    # ------------------------------------------------------------------
    def _invoke_llm(self, prompt: str, **kwargs: Any) -> str:
        if not self._llm_client:
            raise RuntimeError(
                f"LLM client for agent '{self.name}' ({self.llm_service_name}) is not configured."
            )

        print(
            f"Agent '{self.name}': Invoking {self.llm_service_name} with prompt preview: {prompt[:80]!r}"
        )

        try:
            if self.llm_service_name == "gemini":
                gemini_params = kwargs.get("gemini_params", {})
                response = self._llm_client.generate_content(prompt, **gemini_params)
                if (
                    hasattr(response, "candidates")
                    and response.candidates
                    and hasattr(response.candidates[0], "content")
                    and hasattr(response.candidates[0].content, "parts")
                    and response.candidates[0].content.parts
                ):
                    return response.candidates[0].content.parts[0].text
                if hasattr(response, "text") and response.text:
                    return response.text
                if hasattr(response, "parts") and response.parts:
                    return response.parts[0].text
                return ""
            if self.llm_service_name == "openai":
                openai_params = kwargs.get("openai_params", {})
                payload = {
                    "model": "gpt-3.5-turbo",
                    "messages": [{"role": "user", "content": prompt}],
                    **openai_params,
                }
                response = self._llm_client.chat.completions.create(**payload)
                return response.choices[0].message.content
            if self.llm_service_name == "anthropic":
                anthropic_params = kwargs.get("anthropic_params", {})
                payload = {
                    "model": "claude-3-haiku-20240307",
                    "max_tokens": 1024,
                    "messages": [{"role": "user", "content": prompt}],
                    **anthropic_params,
                }
                payload.setdefault("max_tokens", 1024)
                response = self._llm_client.messages.create(**payload)
                return response.content[0].text
        except Exception as exc:  # pragma: no cover - runtime defensive path
            print(
                f"Agent '{self.name}': LLM invocation failed for service {self.llm_service_name}: {exc}"
            )
            return f"Error invoking {self.llm_service_name}: {exc}"

        raise RuntimeError(
            f"LLM invocation not implemented for service '{self.llm_service_name}'."
        )

    # ------------------------------------------------------------------
    # Abstract API expected by orchestrators
    # ------------------------------------------------------------------
    @abstractmethod
    def execute(self, session_state: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        """Produce an update for the shared session state."""

