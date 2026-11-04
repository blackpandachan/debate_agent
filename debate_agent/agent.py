"""Multi-agent debate engine built on the ADK ``Workflow`` graph.

Graph::

    START
      -> init_debate
      -> (pro_stance, con_stance)            # parallel fan-out
      -> collect_stances (JoinNode)          # fan-in
      -> record_stances
      -> pro_debater -> con_debater -> fact_checker -> moderator -> record_round
             ^                                                         |
             +------------------------ "next_round" -------------------+
                                                                       |
                                          "conclude" -> final_judge -> publish_result

The user message is the debate topic. ``max_rounds`` may be preset in session
state; otherwise ``MAX_DEBATE_ROUNDS`` (default 3) is used.

``app`` (an ``App`` with resumability on) is what ``adk web`` / ``adk run`` and
``main.py`` serve; ``root_agent`` is the bare workflow.
"""

from __future__ import annotations

import json
from typing import Any

from google.adk import Agent, Context, Event, Workflow
from google.adk.apps import App, ResumabilityConfig
from google.adk.tools import google_search
from google.adk.workflow import JoinNode, RetryConfig
from google.genai import types

from . import config, prompts
from .evidence import make_evidence_recorder
from .schemas import (
    DebateResult,
    DebateState,
    FinalEvaluation,
    RoundAssessment,
    Stance,
)

APP_NAME = "debate_agent"  # must match this directory's name for `adk web`
NO_TRANSCRIPT = "(No rounds completed yet.)"
TRANSCRIPT_ARTIFACT = "debate_transcript.md"
RESULT_ARTIFACT = "debate_result.json"
EVIDENCE_ARTIFACT = "evidence.json"

# Re-runs a node whose model call or output validation fails (on top of the
# HTTP-level retries configured on the Gemini model).
NODE_RETRY = RetryConfig(max_attempts=3, initial_delay=2.0)


# --------------------------------------------------------------------------
# Function nodes
# --------------------------------------------------------------------------


def init_debate(node_input: str, max_rounds: int = 0):
    """Stores the topic and resets per-debate state."""
    topic = node_input.strip()
    if not topic:
        raise ValueError("Send the debate topic as the message.")
    rounds = max_rounds if max_rounds and max_rounds > 0 else config.max_rounds_default()
    return Event(
        output=f"Prepare your opening position on: {topic}",
        message=f"**Debate topic:** {topic}\n\n**Rounds:** {rounds}",
        state={
            "topic": topic,
            "max_rounds": rounds,
            "current_round": 1,
            "round_scores": [],
            "rounds_md": [],
            "transcript_md": NO_TRANSCRIPT,
            "evidence_log": [],
            "fact_check": "",
        },
    )


def _render_stance(stance: Stance) -> str:
    points = "\n".join(f"- {p}" for p in stance.key_points)
    return f"{stance.position}\n{points}"


def record_stances(node_input: dict[str, Any]):
    """Joins both opening stances and kicks off round 1."""
    pro = Stance.model_validate(node_input["pro_stance"])
    con = Stance.model_validate(node_input["con_stance"])
    pro_text, con_text = _render_stance(pro), _render_stance(con)
    return Event(
        output="Round 1: present your opening argument.",
        message=f"### Opening stances\n\n**PRO:** {pro_text}\n\n**CON:** {con_text}",
        state={"pro_stance_text": pro_text, "con_stance_text": con_text},
    )


def _bullets(title: str, items: list[str]) -> list[str]:
    return ["", f"**{title}:**", *[f"- {i}" for i in items]] if items else []


def _render_round(
    round_num: int,
    pro_argument: str,
    con_argument: str,
    fact_check: str,
    a: RoundAssessment,
) -> str:
    lines = [
        f"## Round {round_num}",
        "",
        "### PRO",
        pro_argument.strip(),
        "",
        "### CON",
        con_argument.strip(),
    ]
    if fact_check.strip():
        lines += ["", "### Fact-check", fact_check.strip()]
    lines += ["", "### Moderator", a.summary.strip()]
    lines += _bullets("Key clashes", a.clashes)
    lines += _bullets("Emerging consensus", a.consensus_points)
    lines += _bullets("Unsupported claims", a.unsupported_claims)
    lines += [
        "",
        f"**Scores:** PRO {a.pro.total}/30 · CON {a.con.total}/30",
        f"- PRO: {a.pro.justification}",
        f"- CON: {a.con.justification}",
    ]
    return "\n".join(lines)


async def record_round(
    ctx: Context,
    node_input: RoundAssessment,
    topic: str,
    current_round: int,
    max_rounds: int,
    pro_argument: str,
    con_argument: str,
    fact_check: str,
    round_scores: list[dict[str, int]],
    rounds_md: list[str],
):
    """Records the round, saves the transcript so far, then loops or concludes."""
    a = node_input
    rounds_md = [
        *rounds_md,
        _render_round(current_round, pro_argument, con_argument, fact_check, a),
    ]
    round_scores = [
        *round_scores,
        {"round": current_round, "pro": a.pro.total, "con": a.con.total},
    ]
    transcript_md = "\n\n".join(rounds_md)
    # Versioned artifact: progress is viewable (in `adk web`) while the
    # debate is still running, and survives a crash.
    await ctx.save_artifact(
        TRANSCRIPT_ARTIFACT,
        types.Part.from_text(text=f"# Debate: {topic}\n\n{transcript_md}"),
    )
    state = {
        "round_scores": round_scores,
        "rounds_md": rounds_md,
        "transcript_md": transcript_md,
        "fact_check": "",
    }
    scored = f"Round {current_round} scored: PRO {a.pro.total}/30, CON {a.con.total}/30."

    if current_round >= max_rounds or a.consensus_reached:
        reason = "consensus reached" if a.consensus_reached else "final round complete"
        return Event(
            output=f"Debate concluded after round {current_round} ({reason}).",
            message=f"{scored} Debate concluded ({reason}).",
            state=state,
            route="conclude",
        )

    next_round = current_round + 1
    state["current_round"] = next_round
    return Event(
        output=f"Round {next_round}: respond to your opponent and advance your case.",
        message=f"{scored} Starting round {next_round}.",
        state=state,
        route="next_round",
    )


def _verdict(pro: float, con: float, outcome: str) -> tuple[str, str]:
    if outcome == "CONSENSUS":
        return "CONSENSUS", "Consensus"
    margin = abs(pro - con)
    if margin < 2.0:
        return "DRAW", "Draw"
    winner = "PRO" if pro > con else "CON"
    return winner, ("Narrow" if margin < 10.0 else "Clear")


def weighted_score(round_totals: list[int], final_total: int) -> float:
    """25% average per-round score (normalized to 100) + 75% final evaluation."""
    if round_totals:
        round_pct = sum(round_totals) / len(round_totals) / 30 * 100
    else:
        round_pct = float(final_total)
    return round(round_pct * 0.25 + final_total * 0.75, 2)


def _evidence_summary(evidence_log: list[dict[str, Any]]) -> list[str]:
    by_speaker: dict[str, tuple[int, int]] = {}
    for entry in evidence_log:
        searches, sources = by_speaker.get(entry["speaker"], (0, 0))
        by_speaker[entry["speaker"]] = (
            searches + len(entry.get("queries", [])),
            sources + len(entry.get("sources", [])),
        )
    if not by_speaker:
        return []
    rows = ["", "## Evidence", "", "| Speaker | Searches | Sources cited |", "|---|---|---|"]
    rows += [f"| {s} | {q} | {n} |" for s, (q, n) in by_speaker.items()]
    return rows


async def publish_result(
    ctx: Context,
    node_input: FinalEvaluation,
    topic: str,
    round_scores: list[dict[str, int]],
    pro_stance_text: str,
    con_stance_text: str,
    transcript_md: str,
    evidence_log: list[dict[str, Any]],
):
    """Aggregates scores, saves artifacts, and emits the final report."""
    e = node_input
    pro_score = weighted_score([r["pro"] for r in round_scores], e.pro.total)
    con_score = weighted_score([r["con"] for r in round_scores], e.con.total)
    winner, victory = _verdict(pro_score, con_score, e.outcome)
    margin = round(abs(pro_score - con_score), 2)

    def row(label: str, field: str) -> str:
        return f"| {label} | {getattr(e.pro, field)} | {getattr(e.con, field)} |"

    report = "\n".join(
        [
            f"# Debate: {topic}",
            "",
            "## Opening stances",
            f"**PRO:** {pro_stance_text}",
            "",
            f"**CON:** {con_stance_text}",
            "",
            transcript_md,
            *_evidence_summary(evidence_log),
            "",
            "## Final evaluation",
            "",
            "| | PRO | CON |",
            "|---|---|---|",
            row("Argument quality (/30)", "argument_quality"),
            row("Rebuttal effectiveness (/25)", "rebuttal_effectiveness"),
            row("Strategic positioning (/20)", "strategic_positioning"),
            row("Rhetorical effectiveness (/15)", "rhetorical_effectiveness"),
            row("Intellectual integrity (/10)", "intellectual_integrity"),
            f"| **Final evaluation (/100)** | **{e.pro.total}** | **{e.con.total}** |",
            f"| **Weighted score** | **{pro_score:.2f}** | **{con_score:.2f}** |",
            "",
            f"**Winner:** {winner} ({victory}, margin {margin:.2f})",
            "",
            e.reasoning.strip(),
        ]
    )
    result = DebateResult(
        topic=topic,
        rounds_completed=len(round_scores),
        pro_weighted_score=pro_score,
        con_weighted_score=con_score,
        winner=winner,
        victory_type=victory,
        margin=margin,
        reasoning=e.reasoning,
        transcript_markdown=report,
    )

    await ctx.save_artifact(TRANSCRIPT_ARTIFACT, types.Part.from_text(text=report))
    await ctx.save_artifact(
        RESULT_ARTIFACT,
        types.Part.from_bytes(
            data=result.model_dump_json(indent=2).encode(), mime_type="application/json"
        ),
    )
    await ctx.save_artifact(
        EVIDENCE_ARTIFACT,
        types.Part.from_bytes(
            data=json.dumps(evidence_log, indent=2).encode(), mime_type="application/json"
        ),
    )

    yield Event(message=report)
    yield Event(
        output=result.model_dump(exclude={"transcript_markdown"}),
        state={
            "final_weighted_score_pro": pro_score,
            "final_weighted_score_con": con_score,
            "winner": winner,
        },
    )


# --------------------------------------------------------------------------
# Graph
# --------------------------------------------------------------------------


def build_root_agent() -> Workflow:
    debate_model_name = config.debate_model_name()
    judge_model_name = config.judge_model_name()
    debate_model = config.build_model(debate_model_name)
    judge_model = config.build_model(judge_model_name)
    with_search = config.search_enabled(debate_model_name)
    with_fact_check = config.fact_check_enabled(judge_model_name)

    def debater(side: str) -> Agent:
        lower = side.lower()
        return Agent(
            name=f"{lower}_debater",
            model=debate_model,
            description=f"Argues the {side} side for one round.",
            instruction=prompts.debater_instruction(side, with_search),
            tools=[google_search] if with_search else [],
            after_model_callback=make_evidence_recorder(side) if with_search else None,
            output_key=f"{lower}_argument",
            retry_config=NODE_RETRY,
        )

    def stance(side: str, instruction: str) -> Agent:
        lower = side.lower()
        return Agent(
            name=f"{lower}_stance",
            model=debate_model,
            description=f"Formulates the opening {side} position.",
            instruction=instruction,
            output_schema=Stance,
            output_key=f"{lower}_stance",
            retry_config=NODE_RETRY,
        )

    pro_stance = stance("PRO", prompts.PRO_STANCE)
    con_stance = stance("CON", prompts.CON_STANCE)
    pro_debater = debater("PRO")
    con_debater = debater("CON")
    moderator = Agent(
        name="moderator",
        model=judge_model,
        description="Summarizes and scores a single debate round.",
        instruction=prompts.MODERATOR,
        output_schema=RoundAssessment,
        output_key="round_assessment",
        retry_config=NODE_RETRY,
    )
    final_judge = Agent(
        name="final_judge",
        model=judge_model,
        description="Holistically scores the full debate and picks an outcome.",
        instruction=prompts.FINAL_JUDGE,
        output_schema=FinalEvaluation,
        output_key="final_evaluation",
        retry_config=NODE_RETRY,
    )

    round_chain: list[Any] = [pro_debater, con_debater]
    if with_fact_check:
        # Plain text output (no output_schema): with search, that is what makes
        # Gemini return grounding metadata, i.e. real source URLs.
        round_chain.append(
            Agent(
                name="fact_checker",
                model=judge_model,
                description="Verifies the round's key factual claims with search.",
                instruction=prompts.FACT_CHECKER,
                tools=[google_search],
                after_model_callback=make_evidence_recorder("FACT_CHECK"),
                output_key="fact_check",
                retry_config=NODE_RETRY,
            )
        )
    round_chain += [moderator, record_round]

    collect_stances = JoinNode(name="collect_stances")
    return Workflow(
        name="debate_workflow",
        description="Two LLM debaters argue a topic over several moderated rounds.",
        state_schema=DebateState,
        edges=[
            ("START", init_debate, (pro_stance, con_stance), collect_stances),
            (collect_stances, record_stances, *round_chain),
            (record_round, {"next_round": pro_debater, "conclude": final_judge}),
            (final_judge, publish_result),
        ],
    )


def build_app(workflow: Workflow | None = None) -> App:
    return App(
        name=APP_NAME,
        root_agent=workflow or build_root_agent(),
        # Persists checkpoints so a debate that fails midway (rate limit,
        # network) can be resumed from the last completed node.
        resumability_config=ResumabilityConfig(is_resumable=True),
    )


root_agent = build_root_agent()
app = build_app(root_agent)
