from __future__ import annotations

from typing import Any, Dict

from google.adk.agents import Agent

from ..core import session_state_contract as contract


class InitializerAgent(Agent):
    """Populate the session state with debate metadata and empty containers."""

    topic: str
    max_rounds: int

    def __init__(self, *, agent_id: str = "initializer_agent", topic: str, max_rounds: int) -> None:
        super().__init__(name=agent_id, topic=topic, max_rounds=max_rounds)

    def execute(self, session_state: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        topic = kwargs.get("debate_topic", self.topic)
        max_rounds = int(kwargs.get("num_rounds", self.max_rounds))
        self.topic = topic
        self.max_rounds = max_rounds

        transcript_header = f"# Debate Topic: {topic}\n\n"

        update = {
            contract.TOPIC: topic,
            contract.MAX_ROUNDS: max_rounds,
            contract.CURRENT_ROUND: 0,
            contract.DEBATE_CONTINUES: True,
            contract.CONSENSUS_REACHED: False,
            contract.CONCESSION_BY: None,
            contract.DEBATE_HISTORY: [],
            contract.PRO_ARGUMENTS_BY_ROUND: {},
            contract.CON_ARGUMENTS_BY_ROUND: {},
            contract.CRITIC_FEEDBACK_BY_ROUND: {},
            contract.ROUND_SCORES: {},
            contract.DEBATE_TRANSCRIPT_MARKDOWN: transcript_header,
        }

        session_state.update(update)
        session_state[contract.LAST_AGENT_STATUS] = {
            "agent": self.name,
            "status": "initialized",
        }
        return session_state

