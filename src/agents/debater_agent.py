from __future__ import annotations

import os
from typing import Any, Dict

from pydantic import PrivateAttr

from ..core import session_state_contract as contract
from .base_llm_agent import BaseLLMAgent


class DebaterAgent(BaseLLMAgent):
    """Produce a structured argument for one side of the debate."""

    _prompt_template: str = PrivateAttr(default="")

    def __init__(
        self,
        *,
        agent_id: str,
        side: str,
        llm_service_name: str,
        prompt_file_path: str,
    ) -> None:
        super().__init__(agent_id=agent_id, llm_service_name=llm_service_name)
        self.side = side.upper()
        if self.side not in {"PRO", "CON"}:
            raise ValueError("DebaterAgent side must be 'PRO' or 'CON'.")
        self._prompt_template = self._load_prompt_template(prompt_file_path)

    @staticmethod
    def _load_prompt_template(file_path: str) -> str:
        resolved_path = os.path.realpath(file_path)
        if not os.path.exists(resolved_path):
            raise FileNotFoundError(f"Prompt file not found: {resolved_path}")
        with open(resolved_path, "r", encoding="utf-8") as handle:
            return handle.read()

    def execute(self, session_state: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        topic = session_state.get(contract.TOPIC, "")
        round_number = session_state.get(contract.CURRENT_ROUND, 0)
        stance = (
            session_state.get(contract.PRO_STANCE)
            if self.side == "PRO"
            else session_state.get(contract.CON_STANCE)
        )
        debate_history = session_state.get(contract.DEBATE_HISTORY, [])
        history_text = "\n".join(
            f"Round {entry.get('round')} {entry.get('speaker')}: {entry.get('content')}"
            for entry in debate_history
        )
        if not history_text:
            history_text = "No prior arguments."

        prompt = (
            f"Debate Topic: {topic}\n"
            f"Your Assigned Stance: {stance or '[Formulate explicitly]'}\n"
            f"Current Round: {round_number}\n"
            f"Debate Context/History:\n{history_text}\n\n"
            f"---\n\n{self._prompt_template}"
        )

        llm_kwargs = kwargs.get("llm_call_kwargs", {})
        response = self.invoke_llm(prompt, **llm_kwargs).strip()

        arguments_key = (
            contract.PRO_ARGUMENTS_BY_ROUND
            if self.side == "PRO"
            else contract.CON_ARGUMENTS_BY_ROUND
        )

        updated_arguments = dict(session_state.get(arguments_key, {}))
        updated_arguments[str(round_number)] = response

        updated_history = list(debate_history)
        updated_history.append(
            {
                "round": round_number,
                "speaker": f"{self.side} Debater",
                "content": response,
            }
        )

        transcript = session_state.get(contract.DEBATE_TRANSCRIPT_MARKDOWN, "")
        transcript += (
            f"## Round {round_number} - {self.side} Debater\n\n{response}\n\n"
        )

        update: Dict[str, Any] = {
            arguments_key: updated_arguments,
            contract.DEBATE_HISTORY: updated_history,
            contract.DEBATE_TRANSCRIPT_MARKDOWN: transcript,
        }

        session_state.update(update)
        session_state[contract.LAST_AGENT_STATUS] = {
            "agent": self.name,
            "status": "argued",
            "round": round_number,
            "side": self.side,
        }
        return session_state

