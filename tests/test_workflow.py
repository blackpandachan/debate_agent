"""Offline tests for the debate workflow using a fake model (no API key needed).

The fake model answers by role, and the debaters' replies carry Google Search
grounding metadata, so the citation/evidence path is exercised too.
"""

from __future__ import annotations

import json

import pytest
from google.adk import Runner
from google.adk.artifacts import InMemoryArtifactService
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.adk.sessions import InMemorySessionService
from google.adk.workflow import RetryConfig
from google.genai import types

from debate_agent import agent as debate
from debate_agent import config
from debate_agent.evidence import add_citations


def _round_assessment(pro: int, con: int, consensus: bool = False) -> dict:
    def side(score: int) -> dict:
        return {
            "argument_quality": score,
            "rebuttal_effectiveness": score,
            "strategic_positioning": score,
            "justification": f"scored {score}",
        }

    return {
        "summary": "Both sides argued with relish.",
        "clashes": ["Does a single hinged bun count as two slices of bread?"],
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

    return {
        "pro": side(28),
        "con": side(18),
        "outcome": outcome,
        "reasoning": "PRO cut the mustard.",
    }


PRO_CLAIM = "Merriam-Webster says a hot dog is a sandwich."


def _grounded(text: str, first_sentence: str) -> types.GroundingMetadata:
    """Grounding metadata citing two pages for the first sentence of text."""
    return types.GroundingMetadata(
        web_search_queries=["is a hot dog a sandwich"],
        grounding_chunks=[
            types.GroundingChunk(
                web=types.GroundingChunkWeb(uri="https://example.com/mw", title="merriam-webster.com")
            ),
            types.GroundingChunk(
                web=types.GroundingChunkWeb(uri="https://example.com/nhdsc", title="hot-dog.org")
            ),
        ],
        grounding_supports=[
            types.GroundingSupport(
                segment=types.Segment(
                    start_index=0, end_index=len(first_sentence.encode()), text=first_sentence
                ),
                grounding_chunk_indices=[0, 1],
            )
        ],
    )


class FakeDebateLlm(BaseLlm):
    """Answers by role, recognised from the system instruction."""

    def __init__(self, *, consensus_on_round: int | None = None, fail_moderator_call: int | None = None):
        # A "gemini-" name lets the built-in google_search tool attach itself.
        super().__init__(model="gemini-fake")
        self._consensus_on_round = consensus_on_round
        self._fail_moderator_call = fail_moderator_call
        self._moderator_calls = 0
        self._instructions: dict[str, list[str]] = {}

    @property
    def instructions(self) -> dict[str, list[str]]:
        return self._instructions

    def stop_failing(self) -> None:
        self._fail_moderator_call = None

    async def generate_content_async(self, llm_request, stream=False):
        system = str(llm_request.config.system_instruction or "")
        grounding = None
        if "preparing the PRO case" in system:
            role, text = "pro_stance", json.dumps(
                {"position": "A hot dog is a sandwich.", "key_points": ["bread + filling"]}
            )
        elif "preparing the CON case" in system:
            role, text = "con_stance", json.dumps(
                {"position": "A hot dog is its own thing.", "key_points": ["cultural identity"]}
            )
        elif "arguing the PRO side" in system:
            role, text = "pro_debater", f"{PRO_CLAIM} Bread hugs meat."
            grounding = _grounded(text, PRO_CLAIM)
        elif "arguing the CON side" in system:
            role, text = "con_debater", "Nobody orders a 'sausage sandwich' at a ballgame."
        elif "independent fact-checker" in system:
            role, text = "fact_checker", '- **PRO** - "Merriam-Webster..." - **Supported** - it really did.'
        elif "debate moderator" in system:
            self._moderator_calls += 1
            if self._moderator_calls == self._fail_moderator_call:
                raise RuntimeError("429 RESOURCE_EXHAUSTED (simulated)")
            consensus = self._moderator_calls == self._consensus_on_round
            role, text = "moderator", json.dumps(_round_assessment(8, 6, consensus))
        elif "final judge" in system:
            role, text = "final_judge", json.dumps(_final_eval())
        else:
            raise AssertionError(f"Unrecognised agent instruction: {system[:200]}")
        self._instructions.setdefault(role, []).append(system)
        yield LlmResponse(
            content=types.Content(role="model", parts=[types.Part(text=text)]),
            grounding_metadata=grounding,
        )


def _build(fake: FakeDebateLlm, monkeypatch, *, search: bool = True):
    monkeypatch.setenv("ENABLE_SEARCH", "true" if search else "false")
    monkeypatch.setenv("ENABLE_FACT_CHECK", "true")
    monkeypatch.setattr(config, "build_model", lambda name: fake)
    monkeypatch.setattr(debate, "NODE_RETRY", RetryConfig(max_attempts=1))
    return Runner(
        app=debate.build_app(debate.build_root_agent()),
        session_service=InMemorySessionService(),
        artifact_service=InMemoryArtifactService(),
    )


async def _start(runner: Runner, topic: str, max_rounds: int):
    session = await runner.session_service.create_session(
        app_name=debate.APP_NAME, user_id="u1", state={"max_rounds": max_rounds}
    )
    msg = types.Content(role="user", parts=[types.Part(text=topic)])
    events = [
        e
        async for e in runner.run_async(user_id="u1", session_id=session.id, new_message=msg)
    ]
    return session.id, events


async def _state(runner: Runner, session_id: str) -> dict:
    session = await runner.session_service.get_session(
        app_name=debate.APP_NAME, user_id="u1", session_id=session_id
    )
    return session.state


async def _artifact(runner: Runner, session_id: str, name: str) -> types.Part | None:
    return await runner.artifact_service.load_artifact(
        app_name=debate.APP_NAME, user_id="u1", session_id=session_id, filename=name
    )


async def test_hot_dog_debate_runs_all_rounds_with_evidence(monkeypatch):
    fake = FakeDebateLlm()
    runner = _build(fake, monkeypatch)
    session_id, _ = await _start(runner, "Is a hot dog a sandwich?", 2)

    assert {k: len(v) for k, v in fake.instructions.items()} == {
        "pro_stance": 1,
        "con_stance": 1,
        "pro_debater": 2,
        "con_debater": 2,
        "fact_checker": 2,
        "moderator": 2,
        "final_judge": 1,
    }

    state = await _state(runner, session_id)
    assert state["topic"] == "Is a hot dog a sandwich?"
    assert state["round_scores"] == [
        {"round": 1, "pro": 24, "con": 18},
        {"round": 2, "pro": 24, "con": 18},
    ]
    # Round avg 24/30 -> 80 * .25 + 83 * .75 = 82.25 ; 60 * .25 + 73 * .75 = 69.75
    assert state["final_weighted_score_pro"] == pytest.approx(82.25)
    assert state["final_weighted_score_con"] == pytest.approx(69.75)
    assert state["winner"] == "PRO"

    # Grounding became inline citations plus a sources list...
    assert state["pro_argument"].startswith(f"{PRO_CLAIM}[1][2] Bread hugs meat.")
    assert "1. [merriam-webster.com](https://example.com/mw)" in state["pro_argument"]
    # ...and an evidence log entry per grounded turn.
    assert [(e["round"], e["speaker"]) for e in state["evidence_log"]] == [(1, "PRO"), (2, "PRO")]

    # State placeholders were resolved into the prompts.
    round_two = fake.instructions["pro_debater"][1]
    assert "Round 2 of 2" in round_two
    assert "A hot dog is a sandwich." in round_two
    assert "### Fact-check" in round_two  # round 1 transcript, fact-check included
    assert "Nobody orders" in fake.instructions["fact_checker"][0]
    assert "**Supported**" in fake.instructions["moderator"][0]

    transcript = await _artifact(runner, session_id, debate.TRANSCRIPT_ARTIFACT)
    assert "**Winner:** PRO (Clear" in transcript.text
    assert "| PRO | 2 | 4 |" in transcript.text  # evidence table: 2 searches, 4 sources
    result = json.loads((await _artifact(runner, session_id, debate.RESULT_ARTIFACT)).inline_data.data)
    assert result["winner"] == "PRO" and result["rounds_completed"] == 2
    evidence = json.loads((await _artifact(runner, session_id, debate.EVIDENCE_ARTIFACT)).inline_data.data)
    assert evidence[0]["sources"][1]["title"] == "hot-dog.org"


async def test_cereal_soup_consensus_ends_debate_early(monkeypatch):
    fake = FakeDebateLlm(consensus_on_round=1)
    runner = _build(fake, monkeypatch)
    session_id, _ = await _start(runner, "Is cereal a soup?", 3)

    state = await _state(runner, session_id)
    assert len(fake.instructions["moderator"]) == 1
    assert len(state["round_scores"]) == 1
    assert len(fake.instructions["final_judge"]) == 1


async def test_pineapple_pizza_without_search_skips_fact_check(monkeypatch):
    fake = FakeDebateLlm()
    runner = _build(fake, monkeypatch, search=False)
    session_id, _ = await _start(runner, "Does pineapple belong on pizza?", 1)

    assert "fact_checker" not in fake.instructions
    assert "google_search" not in fake.instructions["pro_debater"][0]
    state = await _state(runner, session_id)
    assert state["winner"] == "PRO"
    assert state["evidence_log"] == []


async def test_pop_tart_ravioli_debate_resumes_after_failure(monkeypatch):
    """A crash in round 2 resumes without redoing round 1."""
    fake = FakeDebateLlm(fail_moderator_call=2)
    runner = _build(fake, monkeypatch)
    with pytest.raises(RuntimeError, match="simulated"):
        await _start(runner, "Is a Pop-Tart a ravioli?", 2)

    session = (await runner.session_service.list_sessions(app_name=debate.APP_NAME, user_id="u1")).sessions[0]
    full = await runner.session_service.get_session(
        app_name=debate.APP_NAME, user_id="u1", session_id=session.id
    )
    assert len(full.state["round_scores"]) == 1  # round 1 was recorded
    assert "winner" not in full.state or not full.state["winner"]
    invocation_id = full.events[-1].invocation_id

    fake.stop_failing()
    [
        e
        async for e in runner.run_async(
            user_id="u1", session_id=session.id, invocation_id=invocation_id
        )
    ]

    state = await _state(runner, session.id)
    assert state["winner"] == "PRO"
    assert len(state["round_scores"]) == 2
    # Completed nodes were fast-forwarded, not re-run.
    assert len(fake.instructions["pro_stance"]) == 1
    assert len(fake.instructions["pro_debater"]) == 2
    assert len(fake.instructions["final_judge"]) == 1


def test_add_citations_uses_utf8_byte_offsets():
    text = "Würstchen 🌭 are sandwiches. Mustard optional."
    first = "Würstchen 🌭 are sandwiches."
    metadata = _grounded(text, first)
    cited = add_citations(text, metadata, {0: 1, 1: 2})
    assert cited == "Würstchen 🌭 are sandwiches.[1][2] Mustard optional."


def test_verdict_bands():
    assert debate._verdict(80.0, 79.0, "PRO") == ("DRAW", "Draw")
    assert debate._verdict(80.0, 75.0, "PRO") == ("PRO", "Narrow")
    assert debate._verdict(60.0, 80.0, "CON") == ("CON", "Clear")
    assert debate._verdict(60.0, 80.0, "CONSENSUS") == ("CONSENSUS", "Consensus")
