from __future__ import annotations

import json
import os
from typing import Any, Dict

from pydantic import PrivateAttr

from ..core import session_state_contract as contract
from .base_llm_agent import BaseLLMAgent


class CriticAgent(BaseLLMAgent):
    """Evaluates each round and determines whether the debate should continue."""

    _prompt_template: str = PrivateAttr(default="")

    def __init__(
        self,
        *,
        agent_id: str,
        llm_service_name: str,
        prompt_file_path: str,
    ) -> None:
        super().__init__(agent_id=agent_id, llm_service_name=llm_service_name)
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

        pro_argument = session_state.get(contract.PRO_ARGUMENTS_BY_ROUND, {}).get(
            str(round_number),
            "Pro debater did not submit an argument this round.",
        )
        con_argument = session_state.get(contract.CON_ARGUMENTS_BY_ROUND, {}).get(
            str(round_number),
            "Con debater did not submit an argument this round.",
        )

        recent_history = session_state.get(contract.DEBATE_HISTORY, [])[-4:]
        history_text = "\n".join(
            f"Round {entry.get('round')} {entry.get('speaker')}: {entry.get('content')}"
            for entry in recent_history
        ) or "No additional context."

        prompt = self._prompt_template.format(
            topic=topic,
            round_number=round_number,
            pro_argument=pro_argument,
            con_argument=con_argument,
            recent_history=history_text,
        )

        llm_kwargs = kwargs.get("llm_call_kwargs", {})
        raw_response = self.invoke_llm(prompt, **llm_kwargs)

        try:
            parsed = json.loads(raw_response)
        except json.JSONDecodeError:
            parsed = {
                "continue_debate": True,
                "consensus": False,
                "concession": None,
                "feedback": raw_response.strip(),
            }

        feedback_by_round = dict(session_state.get(contract.CRITIC_FEEDBACK_BY_ROUND, {}))
        feedback_by_round[str(round_number)] = parsed

        session_state.update({
            contract.CRITIC_FEEDBACK_BY_ROUND: feedback_by_round,
            contract.DEBATE_CONTINUES: bool(parsed.get("continue_debate", True)),
            contract.CONSENSUS_REACHED: bool(parsed.get("consensus", False)),
            contract.CONCESSION_BY: parsed.get("concession"),
        })

        feedback_text = parsed.get("feedback", "No critic feedback provided.")
        transcript = session_state.get(contract.DEBATE_TRANSCRIPT_MARKDOWN, "")
        transcript += (
            f"### Critic Review - Round {round_number}\n\n{feedback_text}\n\n"
        )
        session_state[contract.DEBATE_TRANSCRIPT_MARKDOWN] = transcript

        history = list(session_state.get(contract.DEBATE_HISTORY, []))
        history.append(
            {
                "round": round_number,
                "speaker": "Critic",
                "content": feedback_text,
            }
        )
        session_state[contract.DEBATE_HISTORY] = history

        session_state[contract.LAST_AGENT_STATUS] = {
            "agent": self.name,
            "status": "reviewed",
            "round": round_number,
        }
        return session_state

