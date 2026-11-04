"""Offline tests for the debate workflow using a fake model (no API key needed)."""

from __future__ import annotations

import json

import pytest
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import InMemoryRunner
from google.genai import types

from debate_agent import agent as debate
from debate_agent import config


def _round_assessment(pro: int, con: int, consensus: bool = False) -> dict:
    def side(score: int) -> dict:
        return {
            "argument_quality": score,
            "rebuttal_effectiveness": score,
            "strategic_positioning": score,
            "justification": f"scored {score}",
        }

    return {
        "summary": "Both sides argued.",
        "clashes": ["safety vs. openness"],
        "consensus_points": [],
        "unsupported_claims": [],
        "pro": side(pro),
        "con": side(con),
        "consensus_reached": consensus,
    }


def _final_eval(outcome: str = "PRO") -> dict:
    def side(aq: int) -> dict:
        return {
            "argument_quality": aq,
            "rebuttal_effectiveness": 20,
            "strategic_positioning": 15,
            "rhetorical_effectiveness": 12,
            "intellectual_integrity": 8,
            "assessment": "solid",
        }

    return {"pro": side(28), "con": side(18), "outcome": outcome, "reasoning": "PRO wins."}


class FakeDebateLlm(BaseLlm):
    """Answers by role, recognised from the system instruction."""

    def __init__(self, *, consensus_on_round: int | None = None):
        super().__init__(model="fake-debate-model")
        self._consensus_on_round = consensus_on_round
        self._moderator_calls = 0
        self._instructions: dict[str, list[str]] = {}

    @property
    def instructions(self) -> dict[str, list[str]]:
        return self._instructions

    async def generate_content_async(self, llm_request, stream=False):
        system = str(llm_request.config.system_instruction or "")
        if "preparing the PRO case" in system:
            role, text = "pro_stance", json.dumps(
                {"position": "Open source is better.", "key_points": ["transparency"]}
            )
        elif "preparing the CON case" in system:
            role, text = "con_stance", json.dumps(
                {"position": "Open source is risky.", "key_points": ["misuse"]}
            )
        elif "arguing the PRO side" in system:
            role, text = "pro_debater", "PRO argument"
        elif "arguing the CON side" in system:
            role, text = "con_debater", "CON argument"
        elif "debate moderator" in system:
            self._moderator_calls += 1
            consensus = self._moderator_calls == self._consensus_on_round
            role, text = "moderator", json.dumps(_round_assessment(8, 6, consensus))
        elif "final judge" in system:
            role, text = "final_judge", json.dumps(_final_eval())
        else:
            raise AssertionError(f"Unrecognised agent instruction: {system[:200]}")
        self._instructions.setdefault(role, []).append(system)
        yield LlmResponse(
            content=types.Content(role="model", parts=[types.Part(text=text)])
        )


async def _run(fake: FakeDebateLlm, monkeypatch, max_rounds: int, app_name: str):
    monkeypatch.setenv("ENABLE_SEARCH", "false")
    monkeypatch.setattr(config, "build_model", lambda name: fake)
    workflow = debate.build_root_agent()

    runner = InMemoryRunner(agent=workflow, app_name=app_name)
    session = await runner.session_service.create_session(
        app_name=app_name, user_id="u1", state={"max_rounds": max_rounds}
    )
    msg = types.Content(role="user", parts=[types.Part(text="Open-source AI?")])
    events = [
        e
        async for e in runner.run_async(
            user_id="u1", session_id=session.id, new_message=msg
        )
    ]
    final = await runner.session_service.get_session(
        app_name=app_name, user_id="u1", session_id=session.id
    )
    return runner, final, events


async def test_full_debate_runs_all_rounds(monkeypatch, request):
    fake = FakeDebateLlm()
    runner, session, _ = await _run(fake, monkeypatch, 2, request.node.name)

    assert len(fake.instructions["pro_debater"]) == 2
    assert len(fake.instructions["moderator"]) == 2
    assert len(fake.instructions["final_judge"]) == 1

    state = session.state
    assert state["topic"] == "Open-source AI?"
    assert [r["round"] for r in state["round_scores"]] == [1, 2]
    assert state["round_scores"][0] == {"round": 1, "pro": 24, "con": 18}
    # Round avg 24/30 -> 80 * .25 + 83 * .75 = 82.25 ; 60 * .25 + 73 * .75 = 69.75
    assert state["final_weighted_score_pro"] == pytest.approx(82.25)
    assert state["final_weighted_score_con"] == pytest.approx(69.75)
    assert state["winner"] == "PRO"

    # State placeholders were resolved into the prompts.
    round_two_prompt = fake.instructions["pro_debater"][1]
    assert "Round 2 of 2" in round_two_prompt
    assert "Open source is better." in round_two_prompt
    assert "## Round 1" in round_two_prompt
    assert "PRO argument" in fake.instructions["con_debater"][0]

    artifact = await runner.artifact_service.load_artifact(
        app_name=request.node.name,
        user_id="u1",
        session_id=session.id,
        filename=debate.TRANSCRIPT_ARTIFACT,
    )
    assert "**Winner:** PRO (Clear" in artifact.text


async def test_consensus_ends_debate_early(monkeypatch, request):
    fake = FakeDebateLlm(consensus_on_round=1)
    _, session, _ = await _run(fake, monkeypatch, 3, request.node.name)

    assert len(fake.instructions["moderator"]) == 1
    assert len(session.state["round_scores"]) == 1
    assert len(fake.instructions["final_judge"]) == 1


def test_verdict_bands():
    assert debate._verdict(80.0, 79.0, "PRO") == ("DRAW", "Draw")
    assert debate._verdict(80.0, 75.0, "PRO") == ("PRO", "Narrow")
    assert debate._verdict(60.0, 80.0, "CON") == ("CON", "Clear")
    assert debate._verdict(60.0, 80.0, "CONSENSUS") == ("CONSENSUS", "Consensus")
