"""
Defines the contract for SessionState keys used in the ADK Multi-Agent Debate Engine.
Each key's purpose, data type, producer agent(s), and consumer agent(s) are documented.
"""

# ---- General Debate Setup ----
TOPIC = "topic"  # (str) The main subject of the debate.
                 # Producer: InitializerAgent
                 # Consumer: ProStanceAgent, ConStanceAgent, ProDebaterAgent, ConDebaterAgent, ModeratorAgent, FinalEvaluatorAgent

MAX_ROUNDS = "max_rounds"  # (int) Total number of debate rounds.
                       # Producer: InitializerAgent
                       # Consumer: ProDebaterAgent, ConDebaterAgent, ModeratorAgent, FinalEvaluatorAgent, MainDebatePipeline (Orchestrator)

CURRENT_ROUND = "current_round"  # (int) The current round number (e.g., 0-indexed or 1-indexed).
                             # Producer: InitializerAgent, MainDebatePipeline (Orchestrator)
                             # Consumer: ProDebaterAgent, ConDebaterAgent, ModeratorAgent

# ---- Stance Formulation ----
PRO_STANCE = "pro_stance"  # (str) The initial affirmative stance.
                       # Producer: ProStanceAgent
                       # Consumer: BranchAgent (Conditional Debate Start), ProDebaterAgent, EndDebateEarlyAgent

PRO_STANCE_STATUS = "pro_stance_status"  # (str) Status of pro stance formulation (e.g., "formulated", "error").
                                     # Producer: ProStanceAgent
                                     # Consumer: (Debugging/Logging)

CON_STANCE = "con_stance"  # (str) The initial opposing stance.
                       # Producer: ConStanceAgent
                       # Consumer: BranchAgent (Conditional Debate Start), ConDebaterAgent, EndDebateEarlyAgent

CON_STANCE_STATUS = "con_stance_status"  # (str) Status of con stance formulation.
                                     # Producer: ConStanceAgent
                                     # Consumer: (Debugging/Logging)

# ---- Debate Execution ----
DEBATE_HISTORY = "debate_history"  # (list of dicts/str) A chronological record of arguments and summaries.
                               # Each entry could be {'round': N, 'speaker': 'Pro/Con/Moderator', 'content': '...'}.
                               # Producer: ProDebaterAgent, ConDebaterAgent, ModeratorAgent (appends to it)
                               # Consumer: ProDebaterAgent, ConDebaterAgent, ModeratorAgent, FinalEvaluatorAgent

PRO_ARGUMENTS_BY_ROUND = "pro_arguments_by_round"  # (dict) {round_num: argument_str} for the Pro debater.
                                               # Producer: ProDebaterAgent
                                               # Consumer: ConDebaterAgent, ModeratorAgent

CON_ARGUMENTS_BY_ROUND = "con_arguments_by_round"  # (dict) {round_num: argument_str} for the Con debater.
                                               # Producer: ConDebaterAgent
                                               # Consumer: ProDebaterAgent, ModeratorAgent

# ---- Moderation & Scoring (Per Round) ----
ROUND_SCORES = "round_scores"  # (dict) {round_num: {'pro_score': float, 'con_score': float, 'summary': str, 
                             #                  'pro_justification': str, 'con_justification': str}}.
                             # Producer: ModeratorAgent
                             # Consumer: FinalEvaluatorAgent, ScoreAggregationAgent

# ---- Final Evaluation & Outcome ----
FINAL_EVALUATION_PRO = "final_evaluation_pro"  # (dict) {'score': float, 'assessment': str}.
                                           # Producer: FinalEvaluatorAgent
                                           # Consumer: ScoreAggregationAgent

FINAL_EVALUATION_CON = "final_evaluation_con"  # (dict) {'score': float, 'assessment': str}.
                                           # Producer: FinalEvaluatorAgent
                                           # Consumer: ScoreAggregationAgent

WINNER_DETERMINATION = "winner_determination"  # (str) "PRO", "CON", or "DRAW".
                                           # Producer: FinalEvaluatorAgent
                                           # Consumer: (Output/Logging)

FINAL_REASONING = "final_reasoning"  # (str) Justification for the winner determination.
                                 # Producer: FinalEvaluatorAgent
                                 # Consumer: (Output/Logging)

FINAL_WEIGHTED_SCORE_PRO = "final_weighted_score_pro"  # (float) Final calculated score for Pro.
                                                   # Producer: ScoreAggregationAgent
                                                   # Consumer: (Output/Logging)

FINAL_WEIGHTED_SCORE_CON = "final_weighted_score_con"  # (float) Final calculated score for Con.
                                                   # Producer: ScoreAggregationAgent
                                                   # Consumer: (Output/Logging)

DEBATE_OUTCOME = "debate_outcome"  # (str) Overall status (e.g., "Completed", "No contest: Stances too similar", "Error: ...").
                               # Producer: EndDebateEarlyAgent, MainDebatePipeline (Orchestrator on error)
                               # Consumer: (Output/Logging)

# ---- Error Handling & Status ----
ERROR_MESSAGE = "error_message"  # (str, optional) Detailed error message if an agent encounters a critical failure.
                             # Producer: Any agent encountering an error
                             # Consumer: (Error handling logic, Logging)

LAST_AGENT_STATUS = "last_agent_status"  # (dict, optional) {agent_name: "completed/failed", "timestamp": ...} for debugging.
                                   # Producer: Each agent could potentially update this for fine-grained status.
                                   # Consumer: (Debugging/Logging)

# ---- Transcript / Logging ----
DEBATE_TRANSCRIPT_MARKDOWN = "debate_transcript_markdown"  # (str) Full debate transcript in Markdown format.
                                                  # Producer: InitializerAgent (initial setup), all other agents append to it.
                                                  # Consumer: (Main script for file output)

# ---- Tool Related State (Example if Search Tool directly puts into SessionState) ----
# Note: Proposal suggests tools return to agent, agent updates SessionState. This is a placeholder if a different pattern is chosen.
# SEARCH_FINDINGS_PRO_ROUND_X = "search_findings_pro_round_X" # (str or list) Search results for Pro in a given round.
# SEARCH_FINDINGS_CON_ROUND_X = "search_findings_con_round_X" # (str or list) Search results for Con in a given round.
