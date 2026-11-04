"""Typed data passed between debate workflow nodes.

ADK validates LLM output against these models (``output_schema``) and
``FunctionNode`` coerces incoming dicts back into them from type hints.
"""

from typing import Literal

from pydantic import BaseModel, Field


class Stance(BaseModel):
    """Opening position for one side of the debate."""

    position: str = Field(
        description="One or two sentences stating the side's thesis on the topic."
    )
    key_points: list[str] = Field(
        description="Two to four core arguments this side intends to develop."
    )


class SideRoundScore(BaseModel):
    """Per-round rubric for one debater (each metric 0-10, max 30)."""

    argument_quality: int = Field(
        ge=0,
        le=10,
        description="Clarity, logic, relevance, and evidence quality this round.",
    )
    rebuttal_effectiveness: int = Field(
        ge=0,
        le=10,
        description="How well the opponent's latest points were addressed.",
    )
    strategic_positioning: int = Field(
        ge=0,
        le=10,
        description="How well the debater advanced their own case.",
    )
    justification: str = Field(description="Brief reasoning for these scores.")

    @property
    def total(self) -> int:
        return (
            self.argument_quality
            + self.rebuttal_effectiveness
            + self.strategic_positioning
        )


class RoundAssessment(BaseModel):
    """The moderator's structured verdict on a single round."""

    summary: str = Field(description="Neutral summary of both sides' arguments.")
    clashes: list[str] = Field(
        description="The main points of direct disagreement this round."
    )
    consensus_points: list[str] = Field(
        default_factory=list,
        description="Points both sides now appear to agree on, if any.",
    )
    unsupported_claims: list[str] = Field(
        default_factory=list,
        description="Factual claims made without credible support, if any.",
    )
    pro: SideRoundScore
    con: SideRoundScore
    consensus_reached: bool = Field(
        description=(
            "True only if the debaters have substantively converged and further"
            " rounds would add nothing."
        )
    )


class SideFinalScore(BaseModel):
    """Holistic 100-point rubric for one debater across the whole debate."""

    argument_quality: int = Field(ge=0, le=30)
    rebuttal_effectiveness: int = Field(ge=0, le=25)
    strategic_positioning: int = Field(ge=0, le=20)
    rhetorical_effectiveness: int = Field(ge=0, le=15)
    intellectual_integrity: int = Field(ge=0, le=10)
    assessment: str = Field(description="Justification for these category scores.")

    @property
    def total(self) -> int:
        return (
            self.argument_quality
            + self.rebuttal_effectiveness
            + self.strategic_positioning
            + self.rhetorical_effectiveness
            + self.intellectual_integrity
        )


class FinalEvaluation(BaseModel):
    """The final judge's evaluation of the complete debate."""

    pro: SideFinalScore
    con: SideFinalScore
    outcome: Literal["PRO", "CON", "CONSENSUS"] = Field(
        description=(
            "Which side made the more persuasive case, or CONSENSUS if there was"
            " no genuine remaining disagreement."
        )
    )
    reasoning: str = Field(description="200-300 word explanation of the outcome.")


class DebateResult(BaseModel):
    """Final output of the workflow."""

    topic: str
    rounds_completed: int
    pro_weighted_score: float
    con_weighted_score: float
    winner: Literal["PRO", "CON", "DRAW", "CONSENSUS"]
    victory_type: Literal["Clear", "Narrow", "Draw", "Consensus"]
    margin: float
    reasoning: str
    transcript_markdown: str
