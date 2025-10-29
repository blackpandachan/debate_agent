"""Session state key constants used across the debate workflow."""

# ---- General Debate Setup ----
TOPIC = "topic"  # (str) The main subject of the debate.
MAX_ROUNDS = "max_rounds"  # (int) Total number of debate rounds permitted.
CURRENT_ROUND = "current_round"  # (int) 1-indexed round marker.
DEBATE_CONTINUES = "debate_continues"  # (bool) Whether additional rounds should be executed.
CONSENSUS_REACHED = "consensus_reached"  # (bool) Whether the debaters reached agreement.
CONCESSION_BY = "concession_by"  # (str|None) Identifier of the side that conceded, if any.

# ---- Stance Formulation ----
PRO_STANCE = "pro_stance"  # (str) The initial affirmative stance.
CON_STANCE = "con_stance"  # (str) The initial opposing stance.
PRO_STANCE_STATUS = "pro_stance_status"  # (str) Status/debug details for pro stance generation.
CON_STANCE_STATUS = "con_stance_status"  # (str) Status/debug details for con stance generation.

# ---- Debate Execution ----
DEBATE_HISTORY = "debate_history"  # (list[dict]) Chronological record of dialogue events.
PRO_ARGUMENTS_BY_ROUND = "pro_arguments_by_round"  # (dict[str, str]) Stored pro arguments by round id.
CON_ARGUMENTS_BY_ROUND = "con_arguments_by_round"  # (dict[str, str]) Stored con arguments by round id.
CRITIC_FEEDBACK_BY_ROUND = "critic_feedback_by_round"  # (dict[str, dict]) Structured critic commentary per round.

# ---- Moderation & Scoring (Per Round) ----
ROUND_SCORES = "round_scores"  # (dict[str, dict]) Moderator scoring summaries keyed by round id.

# ---- Final Evaluation & Outcome ----
FINAL_EVALUATION_PRO = "final_evaluation_pro"
FINAL_EVALUATION_CON = "final_evaluation_con"
WINNER_DETERMINATION = "winner_determination"
FINAL_REASONING = "final_reasoning"
FINAL_WEIGHTED_SCORE_PRO = "final_weighted_score_pro"
FINAL_WEIGHTED_SCORE_CON = "final_weighted_score_con"
DEBATE_OUTCOME = "debate_outcome"

# ---- Error Handling & Status ----
ERROR_MESSAGE = "error_message"
LAST_AGENT_STATUS = "last_agent_status"

# ---- Transcript / Logging ----
DEBATE_TRANSCRIPT_MARKDOWN = "debate_transcript_markdown"
