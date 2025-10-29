from __future__ import annotations

import json
import os
from typing import Any, Dict

from google.adk.agents import Agent
from pydantic import PrivateAttr

from ..core import session_state_contract as contract
from .base_llm_agent import BaseLLMAgent


class FinalEvaluatorAgent(BaseLLMAgent):
    """Holistic end-of-debate judgement."""

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
        history = session_state.get(contract.DEBATE_HISTORY, [])
        round_scores = session_state.get(contract.ROUND_SCORES, {})
        consensus = session_state.get(contract.CONSENSUS_REACHED, False)
        concession = session_state.get(contract.CONCESSION_BY)

        history_text = "\n".join(
            f"Round {entry.get('round')} {entry.get('speaker')}: {entry.get('content')}"
            for entry in history
        )

        round_summary_lines = []
        for round_id, scores in round_scores.items():
            pro_scores = scores.get("pro_scores", {})
            con_scores = scores.get("con_scores", {})
            round_summary_lines.append(
                (
                    f"Round {round_id}: PRO total {pro_scores.get('argument_quality', 0) + pro_scores.get('rebuttal_effectiveness', 0) + pro_scores.get('strategic_positioning', 0)} / CON total {con_scores.get('argument_quality', 0) + con_scores.get('rebuttal_effectiveness', 0) + con_scores.get('strategic_positioning', 0)}"
                )
            )
        round_summary_text = "\n".join(round_summary_lines) or "No scoring data available."

        prompt = self._prompt_template + (
            "\n\nReturn a JSON object with keys: 'pro', 'con', 'winner', 'reasoning'.\n"
            "- 'pro' and 'con' should be objects with 'score' (0-100) and 'assessment' (string).\n"
            "- 'winner' must be 'PRO', 'CON', or 'DRAW'.\n"
            "- 'reasoning' should be a concise justification summarising the debate outcome.\n"
            "Do not include any additional commentary outside the JSON."
        )

        formatted_prompt = prompt.format(
            topic=topic,
            debate_history=history_text,
            round_scores=round_summary_text,
            consensus=str(consensus),
            concession=concession or "None",
        )

        llm_kwargs = kwargs.get("llm_call_kwargs", {})
        raw_response = self.invoke_llm(formatted_prompt, **llm_kwargs)

        try:
            parsed = json.loads(raw_response)
        except json.JSONDecodeError:
            parsed = {
                "pro": {"score": 0.0, "assessment": raw_response.strip() or "No assessment."},
                "con": {"score": 0.0, "assessment": "Parsing failed."},
                "winner": "DRAW",
                "reasoning": "Unable to parse evaluator response; defaulting to draw.",
            }

        update = {
            contract.FINAL_EVALUATION_PRO: parsed.get("pro", {}),
            contract.FINAL_EVALUATION_CON: parsed.get("con", {}),
            contract.WINNER_DETERMINATION: parsed.get("winner", "DRAW"),
            contract.FINAL_REASONING: parsed.get("reasoning", ""),
        }

        session_state.update(update)
        session_state[contract.LAST_AGENT_STATUS] = {
            "agent": self.name,
            "status": "evaluated",
        }
        return session_state


class ScoreAggregationAgent(Agent):
    """Compute weighted scores based on per-round results and final evaluation."""

    def __init__(self, *, agent_id: str = "score_aggregation_agent") -> None:
        super().__init__(name=agent_id)

    def run(self, session_state: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        round_scores = session_state.get(contract.ROUND_SCORES, {})
        final_pro = session_state.get(contract.FINAL_EVALUATION_PRO, {})
        final_con = session_state.get(contract.FINAL_EVALUATION_CON, {})

        pro_round_totals = []
        con_round_totals = []
        for scores in round_scores.values():
            pro = scores.get("pro_scores", {})
            con = scores.get("con_scores", {})
            pro_round_totals.append(
                float(pro.get("argument_quality", 0))
                + float(pro.get("rebuttal_effectiveness", 0))
                + float(pro.get("strategic_positioning", 0))
            )
            con_round_totals.append(
                float(con.get("argument_quality", 0))
                + float(con.get("rebuttal_effectiveness", 0))
                + float(con.get("strategic_positioning", 0))
            )

        pro_avg = sum(pro_round_totals) / len(pro_round_totals) if pro_round_totals else 0.0
        con_avg = sum(con_round_totals) / len(con_round_totals) if con_round_totals else 0.0

        final_weighted_score_pro = (pro_avg * 0.25) + float(final_pro.get("score", 0)) * 0.75
        final_weighted_score_con = (con_avg * 0.25) + float(final_con.get("score", 0)) * 0.75

        session_state[contract.FINAL_WEIGHTED_SCORE_PRO] = final_weighted_score_pro
        session_state[contract.FINAL_WEIGHTED_SCORE_CON] = final_weighted_score_con

        session_state[contract.LAST_AGENT_STATUS] = {
            "agent": self.name,
            "status": "scores_aggregated",
        }
        return session_state

