from __future__ import annotations

import os
from typing import Any, Dict

from pydantic import PrivateAttr

from ..core import session_state_contract as contract
from .base_llm_agent import BaseLLMAgent


class _StanceAgent(BaseLLMAgent):
    """Shared implementation for stance agents."""

    _prompt_template: str = PrivateAttr(default="")

    def __init__(
        self,
        *,
        agent_id: str,
        llm_service_name: str,
        prompt_file_path: str,
        stance_key: str,
        status_key: str,
    ) -> None:
        super().__init__(agent_id=agent_id, llm_service_name=llm_service_name)
        self._prompt_template = self._load_prompt_template(prompt_file_path)
        self._stance_key = stance_key
        self._status_key = status_key

    @staticmethod
    def _load_prompt_template(file_path: str) -> str:
        resolved_path = os.path.realpath(file_path)
        if not os.path.exists(resolved_path):
            raise FileNotFoundError(f"Prompt file not found: {resolved_path}")
        with open(resolved_path, "r", encoding="utf-8") as handle:
            return handle.read()

    def execute(self, session_state: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        topic = session_state.get(contract.TOPIC, "")
        history_snippets = [
            f"Round {entry.get('round', '?')} {entry.get('speaker', 'Agent')}: {entry.get('content', '')}"
            for entry in session_state.get(contract.DEBATE_HISTORY, [])
        ]
        history_block = "\n".join(history_snippets).strip() or "No debate history yet."

        prompt = self._prompt_template.format(topic=topic, debate_history=history_block)
        response = self.invoke_llm(prompt, **kwargs).strip()

        update = {
            self._stance_key: response,
            self._status_key: "generated" if response else "empty_response",
        }

        transcript = session_state.get(contract.DEBATE_TRANSCRIPT_MARKDOWN, "")
        transcript += (
            f"**{self.name.replace('_', ' ').title()} Stance**\n\n"
            f"{response or 'No stance generated.'}\n\n"
        )
        update[contract.DEBATE_TRANSCRIPT_MARKDOWN] = transcript

        session_state.update(update)
        session_state[contract.LAST_AGENT_STATUS] = {
            "agent": self.name,
            "status": update[self._status_key],
        }
        return session_state


class ProStanceAgent(_StanceAgent):
    def __init__(self, *, agent_id: str, llm_service_name: str, prompt_file_path: str) -> None:
        super().__init__(
            agent_id=agent_id,
            llm_service_name=llm_service_name,
            prompt_file_path=prompt_file_path,
            stance_key=contract.PRO_STANCE,
            status_key=contract.PRO_STANCE_STATUS,
        )


class ConStanceAgent(_StanceAgent):
    def __init__(self, *, agent_id: str, llm_service_name: str, prompt_file_path: str) -> None:
        super().__init__(
            agent_id=agent_id,
            llm_service_name=llm_service_name,
            prompt_file_path=prompt_file_path,
            stance_key=contract.CON_STANCE,
            status_key=contract.CON_STANCE_STATUS,
        )

