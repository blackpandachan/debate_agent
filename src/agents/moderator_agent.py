from __future__ import annotations

import json
import os
from typing import Any, Dict

from pydantic import PrivateAttr

from ..core import session_state_contract as contract
from .base_llm_agent import BaseLLMAgent


class ModeratorAgent(BaseLLMAgent):
    """Summarises each round and issues per-round scores."""

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
        critic_feedback = session_state.get(contract.CRITIC_FEEDBACK_BY_ROUND, {}).get(
            str(round_number),
            {},
        )

        critic_text = critic_feedback.get("feedback", "No critic feedback recorded.")

        moderation_prompt = self._prompt_template + (
            "\n\n"
            "You must produce a single JSON object with the following structure:\n"
            "{\n"
            "  \"summary\": string,\n"
            "  \"pro_scores\": {\n"
            "    \"argument_quality\": number,\n"
            "    \"rebuttal_effectiveness\": number,\n"
            "    \"strategic_positioning\": number,\n"
            "    \"justification\": string\n"
            "  },\n"
            "  \"con_scores\": {\n"
            "    \"argument_quality\": number,\n"
            "    \"rebuttal_effectiveness\": number,\n"
            "    \"strategic_positioning\": number,\n"
            "    \"justification\": string\n"
            "  },\n"
            "  \"round_winner\": \"PRO\" | \"CON\" | \"DRAW\",\n"
            "  \"moderator_notes\": string\n"
            "}\n"
            "Use plain numbers for the scores (0-10). Do not include any text outside of the JSON object."
        )

        prompt = moderation_prompt.format(
            topic=topic,
            round_number=round_number,
            pro_argument=pro_argument,
            con_argument=con_argument,
            critic_feedback=critic_text,
        )

        llm_kwargs = kwargs.get("llm_call_kwargs", {})
        raw_response = self.invoke_llm(prompt, **llm_kwargs)

        try:
            parsed = json.loads(raw_response)
        except json.JSONDecodeError:
            parsed = {
                "summary": raw_response.strip(),
                "pro_scores": {
                    "argument_quality": 0.0,
                    "rebuttal_effectiveness": 0.0,
                    "strategic_positioning": 0.0,
                    "justification": "Unable to parse moderator response.",
                },
                "con_scores": {
                    "argument_quality": 0.0,
                    "rebuttal_effectiveness": 0.0,
                    "strategic_positioning": 0.0,
                    "justification": "Unable to parse moderator response.",
                },
                "round_winner": "DRAW",
                "moderator_notes": "Parsing failure; defaulted values.",
            }

        round_scores = dict(session_state.get(contract.ROUND_SCORES, {}))
        round_scores[str(round_number)] = parsed

        transcript = session_state.get(contract.DEBATE_TRANSCRIPT_MARKDOWN, "")
        transcript += (
            f"### Moderator Summary - Round {round_number}\n\n"
            f"{parsed.get('summary')}\n\n"
            "**Pro Scores**\n"
            f"- Argument Quality: {parsed['pro_scores']['argument_quality']}\n"
            f"- Rebuttal Effectiveness: {parsed['pro_scores']['rebuttal_effectiveness']}\n"
            f"- Strategic Positioning: {parsed['pro_scores']['strategic_positioning']}\n"
            f"- Justification: {parsed['pro_scores']['justification']}\n\n"
            "**Con Scores**\n"
            f"- Argument Quality: {parsed['con_scores']['argument_quality']}\n"
            f"- Rebuttal Effectiveness: {parsed['con_scores']['rebuttal_effectiveness']}\n"
            f"- Strategic Positioning: {parsed['con_scores']['strategic_positioning']}\n"
            f"- Justification: {parsed['con_scores']['justification']}\n\n"
        )

        session_state.update({
            contract.ROUND_SCORES: round_scores,
            contract.DEBATE_TRANSCRIPT_MARKDOWN: transcript,
        })

        history = list(session_state.get(contract.DEBATE_HISTORY, []))
        history.append(
            {
                "round": round_number,
                "speaker": "Moderator",
                "content": parsed.get("summary"),
            }
        )
        session_state[contract.DEBATE_HISTORY] = history

        session_state[contract.LAST_AGENT_STATUS] = {
            "agent": self.name,
            "status": "moderated",
            "round": round_number,
        }
        return session_state

