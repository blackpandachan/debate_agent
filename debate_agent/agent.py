"""Multi-agent debate engine built on the ADK ``Workflow`` graph.

Graph::

    START
      -> init_debate
      -> (pro_stance, con_stance)            # parallel fan-out
      -> collect_stances (JoinNode)          # fan-in
      -> record_stances
      -> pro_debater -> con_debater -> moderator -> record_round
                 ^                                      |
                 +-------------- "next_round" ----------+
                                                        |
                                    "conclude" -> final_judge -> publish_result

The user message is the debate topic. ``max_rounds`` may be preset in session
state; otherwise ``MAX_DEBATE_ROUNDS`` (default 3) is used.

Run it with ``adk web`` / ``adk run debate_agent`` or ``python main.py``.
"""

from __future__ import annotations

from typing import Any

from google.adk import Agent, Context, Event, Workflow
from google.adk.tools import google_search
from google.adk.workflow import JoinNode
from google.genai import types

from . import config, prompts
from .schemas import (
    DebateResult,
    FinalEvaluation,
    RoundAssessment,
    Stance,
)

NO_TRANSCRIPT = "(No rounds completed yet.)"
TRANSCRIPT_ARTIFACT = "debate_transcript.md"


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
            "transcript_md": NO_TRANSCRIPT,
            "rounds_md": [],
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


def _render_round(
    round_num: int, pro_argument: str, con_argument: str, a: RoundAssessment
) -> str:
    lines = [
        f"## Round {round_num}",
        "",
        "### PRO",
        pro_argument.strip(),
        "",
        "### CON",
        con_argument.strip(),
        "",
        "### Moderator",
        a.summary.strip(),
    ]
    if a.clashes:
        lines += ["", "**Key clashes:**", *[f"- {c}" for c in a.clashes]]
    if a.consensus_points:
        lines += ["", "**Emerging consensus:**", *[f"- {c}" for c in a.consensus_points]]
    if a.unsupported_claims:
        lines += ["", "**Unsupported claims:**", *[f"- {c}" for c in a.unsupported_claims]]
    lines += [
        "",
        f"**Scores:** PRO {a.pro.total}/30 · CON {a.con.total}/30",
        f"- PRO: {a.pro.justification}",
        f"- CON: {a.con.justification}",
    ]
    return "\n".join(lines)


def record_round(
    node_input: RoundAssessment,
    current_round: int,
    max_rounds: int,
    pro_argument: str,
    con_argument: str,
    round_scores: list[dict[str, Any]],
    rounds_md: list[str],
):
    """Records the moderator's scores, then loops or concludes."""
    assessment = node_input
    round_md = _render_round(current_round, pro_argument, con_argument, assessment)
    rounds_md = [*rounds_md, round_md]
    round_scores = [
        *round_scores,
        {
            "round": current_round,
            "pro": assessment.pro.total,
            "con": assessment.con.total,
        },
    ]
    state = {
        "round_scores": round_scores,
        "rounds_md": rounds_md,
        "transcript_md": "\n\n".join(rounds_md),
    }

    done = current_round >= max_rounds or assessment.consensus_reached
    if done:
        reason = (
            "consensus reached" if assessment.consensus_reached else "final round complete"
        )
        return Event(
            output=f"Debate concluded after round {current_round} ({reason}).",
            message=(
                f"Round {current_round} scored — PRO {assessment.pro.total}/30,"
                f" CON {assessment.con.total}/30. Debate concluded ({reason})."
            ),
            state=state,
            route="conclude",
        )

    next_round = current_round + 1
    state["current_round"] = next_round
    return Event(
        output=f"Round {next_round}: respond to your opponent and advance your case.",
        message=(
            f"Round {current_round} scored — PRO {assessment.pro.total}/30,"
            f" CON {assessment.con.total}/30. Starting round {next_round}."
        ),
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


async def publish_result(
    ctx: Context,
    node_input: FinalEvaluation,
    topic: str,
    round_scores: list[dict[str, Any]],
    pro_stance_text: str,
    con_stance_text: str,
    transcript_md: str,
):
    """Aggregates scores, saves the transcript artifact, and emits the result."""
    evaluation = node_input
    pro_score = weighted_score([r["pro"] for r in round_scores], evaluation.pro.total)
    con_score = weighted_score([r["con"] for r in round_scores], evaluation.con.total)
    winner, victory = _verdict(pro_score, con_score, evaluation.outcome)

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
            "",
            "## Final evaluation",
            "",
            "| | PRO | CON |",
            "|---|---|---|",
            f"| Argument quality (/30) | {evaluation.pro.argument_quality} | {evaluation.con.argument_quality} |",
            f"| Rebuttal effectiveness (/25) | {evaluation.pro.rebuttal_effectiveness} | {evaluation.con.rebuttal_effectiveness} |",
            f"| Strategic positioning (/20) | {evaluation.pro.strategic_positioning} | {evaluation.con.strategic_positioning} |",
            f"| Rhetorical effectiveness (/15) | {evaluation.pro.rhetorical_effectiveness} | {evaluation.con.rhetorical_effectiveness} |",
            f"| Intellectual integrity (/10) | {evaluation.pro.intellectual_integrity} | {evaluation.con.intellectual_integrity} |",
            f"| **Final evaluation (/100)** | **{evaluation.pro.total}** | **{evaluation.con.total}** |",
            f"| **Weighted score** | **{pro_score:.2f}** | **{con_score:.2f}** |",
            "",
            f"**Winner:** {winner} ({victory}, margin {abs(pro_score - con_score):.2f})",
            "",
            evaluation.reasoning.strip(),
        ]
    )

    await ctx.save_artifact(TRANSCRIPT_ARTIFACT, types.Part.from_text(text=report))

    result = DebateResult(
        topic=topic,
        rounds_completed=len(round_scores),
        pro_weighted_score=pro_score,
        con_weighted_score=con_score,
        winner=winner,
        victory_type=victory,
        margin=round(abs(pro_score - con_score), 2),
        reasoning=evaluation.reasoning,
        transcript_markdown=report,
    )
    yield Event(message=report)
    yield Event(
        output=result.model_dump(),
        state={
            "final_weighted_score_pro": pro_score,
            "final_weighted_score_con": con_score,
            "winner": winner,
        },
    )


# --------------------------------------------------------------------------
# LLM agents
# --------------------------------------------------------------------------


def build_root_agent() -> Workflow:
    debate_model_name = config.debate_model_name()
    judge_model_name = config.judge_model_name()
    debate_model = config.build_model(debate_model_name)
    judge_model = config.build_model(judge_model_name)
    with_search = config.search_enabled(debate_model_name)
    debater_tools = [google_search] if with_search else []

    pro_stance = Agent(
        name="pro_stance",
        model=debate_model,
        description="Formulates the opening PRO position.",
        instruction=prompts.PRO_STANCE,
        output_schema=Stance,
        output_key="pro_stance",
    )
    con_stance = Agent(
        name="con_stance",
        model=debate_model,
        description="Formulates the opening CON position.",
        instruction=prompts.CON_STANCE,
        output_schema=Stance,
        output_key="con_stance",
    )
    pro_debater = Agent(
        name="pro_debater",
        model=debate_model,
        description="Argues the PRO side for one round.",
        instruction=prompts.debater_instruction("PRO", with_search),
        tools=debater_tools,
        output_key="pro_argument",
    )
    con_debater = Agent(
        name="con_debater",
        model=debate_model,
        description="Argues the CON side for one round.",
        instruction=prompts.debater_instruction("CON", with_search),
        tools=debater_tools,
        output_key="con_argument",
    )
    moderator = Agent(
        name="moderator",
        model=judge_model,
        description="Summarizes and scores a single debate round.",
        instruction=prompts.MODERATOR,
        output_schema=RoundAssessment,
        output_key="round_assessment",
    )
    final_judge = Agent(
        name="final_judge",
        model=judge_model,
        description="Holistically scores the full debate and picks an outcome.",
        instruction=prompts.FINAL_JUDGE,
        output_schema=FinalEvaluation,
        output_key="final_evaluation",
    )

    collect_stances = JoinNode(name="collect_stances")

    return Workflow(
        name="debate_workflow",
        description="Two LLM debaters argue a topic over several moderated rounds.",
        edges=[
            ("START", init_debate, (pro_stance, con_stance), collect_stances),
            (collect_stances, record_stances, pro_debater, con_debater, moderator, record_round),
            (record_round, {"next_round": pro_debater, "conclude": final_judge}),
            (final_judge, publish_result),
        ],
    )


root_agent = build_root_agent()
