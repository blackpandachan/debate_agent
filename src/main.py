import os
import sys
import logging
from dotenv import load_dotenv

# Determine Project Root Dynamically and add to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import from google.adk.agents (the ADK 0.5.0 structure)
from google.adk.agents import (
    ParallelAgent,
    SequentialAgent,
    LoopAgent,
    Agent, 
    BaseAgent
)
from google.adk import Runner
from typing import Callable
from google.adk.tools import google_search

# Now that PROJECT_ROOT is in sys.path, these imports should work
from src.agents.initializer_agent import InitializerAgent # Though not used directly in main flow now
from src.agents.stance_agents import ProStanceAgent, ConStanceAgent
from src.agents.debater_agent import DebaterAgent
from src.agents.moderator_agent import ModeratorAgent
from src.agents.evaluator_agents import FinalEvaluatorAgent, ScoreAggregationAgent
from src.core import session_state_contract as contract

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
DEBATE_TOPIC = "Should generative AI models be open-sourced?"
MAX_DEBATE_ROUNDS = 3
LLM_SERVICE = os.getenv("LLM_SERVICE_MAIN", "gemini").lower() # Default to gemini

# PROJECT_ROOT is defined above and in sys.path
DOTENV_PATH = os.path.join(PROJECT_ROOT, '.env')
PROMPTS_DIR = os.path.join(PROJECT_ROOT, "src", "prompts")

# Prompt File Paths


# Prompt File Paths
PRO_STANCE_PROMPT = os.path.join(PROMPTS_DIR, "pro_stance_prompt.txt")
CON_STANCE_PROMPT = os.path.join(PROMPTS_DIR, "con_stance_prompt.txt")
PRO_DEBATER_PROMPT = os.path.join(PROMPTS_DIR, "pro_debater_prompt.txt")
CON_DEBATER_PROMPT = os.path.join(PROMPTS_DIR, "con_debater_prompt.txt")
MODERATOR_PROMPT = os.path.join(PROMPTS_DIR, "moderator_prompt.txt")
FINAL_EVALUATOR_PROMPT = os.path.join(PROMPTS_DIR, "final_evaluator_prompt.txt")

def main():

    logger.info(f"Loading .env from: {DOTENV_PATH}")
    load_dotenv(dotenv_path=DOTENV_PATH)

    logger.info(f"Starting debate on: {DEBATE_TOPIC}")
    logger.info(f"LLM Service in use: {LLM_SERVICE.upper()}")
    MAX_ROUNDS = int(os.getenv("MAX_DEBATE_ROUNDS", "3"))
    logger.info(f"Max debate rounds: {MAX_ROUNDS}")

    # --- Tool Instantiation ---
    # WebSearchTool is used by BaseLLMAgent for Gemini if adk_tools is not specified
    # DebaterAgent for Gemini specifically includes google_search tool
    web_search_tool = google_search # Available if needed by other agents

    # --- Agent Instantiation ---
    initializer_agent = InitializerAgent(
        agent_id="initializer_agent",
        topic=DEBATE_TOPIC,
        max_rounds=MAX_DEBATE_ROUNDS
    )
    
    pro_stance_agent = ProStanceAgent(
        agent_id="pro_stance_agent",
        llm_service_name=LLM_SERVICE,
        prompt_file_path=PRO_STANCE_PROMPT
    )
    
    con_stance_agent = ConStanceAgent(
        agent_id="con_stance_agent",
        llm_service_name=LLM_SERVICE,
        prompt_file_path=CON_STANCE_PROMPT
    )
    
    pro_debater_agent = DebaterAgent(
        agent_id="pro_debater_agent",
        llm_service_name=LLM_SERVICE,
        prompt_file_path=PRO_DEBATER_PROMPT
    )
    
    con_debater_agent = DebaterAgent(
        agent_id="con_debater_agent",
        llm_service_name=LLM_SERVICE,
        prompt_file_path=CON_DEBATER_PROMPT
    )
    moderator_agent = ModeratorAgent(
        agent_id="moderator_agent",
        llm_service_name=LLM_SERVICE,
        prompt_file_path=MODERATOR_PROMPT
    )
    final_evaluator_agent = FinalEvaluatorAgent(
        agent_id="final_evaluator_agent",
        llm_service_name=LLM_SERVICE,
        prompt_file_path=FINAL_EVALUATOR_PROMPT
    )
    score_aggregation_agent = ScoreAggregationAgent(
        name="score_aggregation_agent"
    )

    # --- Orchestration ---
    # Phase 1: Initialization and Stance Formulation
    initial_phase = SequentialAgent(
        name="InitialPhase",
        sub_agents=[
            initializer_agent,
            ParallelAgent(
                name="StanceFormulation",
                sub_agents=[pro_stance_agent, con_stance_agent]
            )
        ]
    )

    # Phase 2: Debate Rounds with Moderator
    # Define a single round of debate
    debate_round_sequence = SequentialAgent(
        name="DebateRound",
        sub_agents=[
            pro_debater_agent, 
            con_debater_agent, 
            moderator_agent
        ]
    )

    # Loop for MAX_DEBATE_ROUNDS
    # Condition: current_round < max_rounds AND debate_continues (from moderator)
    def should_continue_debate(session_state: dict) -> bool:
        current_round = session_state.get(contract.CURRENT_ROUND, 0)
        max_r = session_state.get(contract.MAX_ROUNDS, MAX_DEBATE_ROUNDS)
        debate_continues = session_state.get(contract.DEBATE_CONTINUES, True)
        can_proceed = current_round < max_r and debate_continues
        if not can_proceed:
            logger.info(f"Debate loop condition not met. Current Round: {current_round}, Max Rounds: {max_r}, Debate Continues Flag: {debate_continues}")
        return can_proceed

    debate_loop = LoopAgent(
        name="DebateLoop",
        sub_agents=[debate_round_sequence],
        max_iterations=MAX_DEBATE_ROUNDS # Safety break
    )

    # Phase 3: Final Evaluation and Scoring
    final_phase = SequentialAgent(
        name="FinalPhase",
        sub_agents=[final_evaluator_agent, score_aggregation_agent]
    )

    # Full Debate Pipeline
    debate_pipeline = SequentialAgent(
        name="FullDebatePipeline",
        sub_agents=[
            initial_phase,
            debate_loop, 
            final_phase
        ]
    )

    # --- Execution ---
    # Let's try a simpler approach without using the Runner
    # We'll manually execute the top-level agent and its subagents
    
    logger.info("Executing debate pipeline directly without Runner...")
    
    try:
        # Create a simple dictionary to act as our session state
        final_session_state = {
            contract.TOPIC: DEBATE_TOPIC,
            contract.MAX_ROUNDS: MAX_DEBATE_ROUNDS,
            contract.CURRENT_ROUND: 0,
            # The DEBATE_CONTINUES key seems to be missing from the contract
            # Using it anyway since it appears to be needed by our code
            "debate_continues": True
        }
        
        # Let's call each agent in sequence ourselves
        
        # Phase 1: Setup
        logger.info("Executing initialization phase...")
        logger.info(f"Initializing with topic: {DEBATE_TOPIC}")
        
        # Execute the initializer agent directly
        logger.info("Running initializer agent...")
        try:
            init_result = initializer_agent.execute(final_session_state)
            final_session_state.update(init_result)
            logger.info(f"Initializer result: {init_result}")
        except Exception as init_e:
            logger.error(f"Error in initializer agent: {init_e}")
        
        # Execute the stance agents
        logger.info("Running pro stance agent...")
        try:
            pro_stance_result = pro_stance_agent.execute(final_session_state)
            final_session_state.update(pro_stance_result)
            logger.info(f"Pro stance: {final_session_state.get(contract.PRO_STANCE, 'Not set')}")
        except Exception as pro_e:
            logger.error(f"Error in pro stance agent: {pro_e}")
            
        logger.info("Running con stance agent...")
        try:
            con_stance_result = con_stance_agent.execute(final_session_state)
            final_session_state.update(con_stance_result)
            logger.info(f"Con stance: {final_session_state.get(contract.CON_STANCE, 'Not set')}")
        except Exception as con_e:
            logger.error(f"Error in con stance agent: {con_e}")
        
        # Phase 2: Debate rounds
        for round_num in range(1, MAX_DEBATE_ROUNDS + 1):
            if not final_session_state.get("debate_continues", False):
                logger.info(f"Debate stopped after round {round_num-1}")
                break
                
            logger.info(f"\n--- Starting Debate Round {round_num} ---")
            final_session_state[contract.CURRENT_ROUND] = round_num
            
            # Pro debater
            logger.info("Running pro debater...")
            try:
                pro_result = pro_debater_agent.execute(final_session_state)
                final_session_state.update(pro_result)
                logger.info(f"Pro argument added for round {round_num}")
            except Exception as e:
                logger.error(f"Error in pro debater: {e}")
            
            # Con debater
            logger.info("Running con debater...")
            try:
                con_result = con_debater_agent.execute(final_session_state)
                final_session_state.update(con_result)
                logger.info(f"Con argument added for round {round_num}")
            except Exception as e:
                logger.error(f"Error in con debater: {e}")
            
            # Moderator
            logger.info("Running moderator...")
            try:
                mod_result = moderator_agent.execute(final_session_state)
                final_session_state.update(mod_result)
                logger.info(f"Round {round_num} moderated. Scores - Pro: {final_session_state.get(f'round_{round_num}_score_pro', 'N/A')}, Con: {final_session_state.get(f'round_{round_num}_score_con', 'N/A')}")
            except Exception as e:
                logger.error(f"Error in moderator: {e}")
        
        # Phase 3: Final evaluation
        logger.info("\n--- Final Evaluation ---")
        
        # Final evaluator
        logger.info("Running final evaluator...")
        try:
            eval_result = final_evaluator_agent.execute(final_session_state)
            final_session_state.update(eval_result)
            logger.info(f"Final evaluation complete")
        except Exception as e:
            logger.error(f"Error in final evaluator: {e}")
        
        # Score aggregation
        logger.info("Running score aggregation...")
        try:
            agg_result = score_aggregation_agent.execute(final_session_state)
            final_session_state.update(agg_result)
            logger.info(f"Scores aggregated. Final weighted scores - Pro: {final_session_state.get(contract.FINAL_WEIGHTED_SCORE_PRO, 'N/A')}, Con: {final_session_state.get(contract.FINAL_WEIGHTED_SCORE_CON, 'N/A')}")
        except Exception as e:
            logger.error(f"Error in score aggregation: {e}")
        
        logger.info("Debate pipeline execution complete!")
        
    except Exception as e:
        logger.error(f"An error occurred during direct pipeline execution: {e}")
        logger.error("Check agent __init__ methods to ensure super().__init__ is called correctly with all necessary ADK parameters.")
        import traceback
        traceback.print_exc()
        return # Exit if critical error

    logger.info("\n--- DEBATE RESULTS ---")
    logger.info(f"Topic: {final_session_state.get(contract.TOPIC)}")
    logger.info(f"Max Rounds: {final_session_state.get(contract.MAX_ROUNDS)}")
    logger.info(f"Completed Rounds: {final_session_state.get(contract.CURRENT_ROUND)}")

    logger.info("\n--- Stances ---")
    logger.info(f"Pro Stance: {final_session_state.get(contract.PRO_STANCE)}")
    logger.info(f"Con Stance: {final_session_state.get(contract.CON_STANCE)}")

    logger.info("\n--- Final Scores ---")
    logger.info(f"Pro Final Weighted Score: {final_session_state.get(contract.FINAL_WEIGHTED_SCORE_PRO, 0.0):.2f}")
    logger.info(f"Con Final Weighted Score: {final_session_state.get(contract.FINAL_WEIGHTED_SCORE_CON, 0.0):.2f}")
    logger.info(f"Overall Winner: {final_session_state.get(contract.WINNER_DETERMINATION, 'N/A')}")
    logger.info(f"Reasoning: {final_session_state.get(contract.FINAL_REASONING, 'N/A')}")

    logger.info("\n--- Full Debate Transcript (Markdown) ---")
    transcript_md = final_session_state.get(contract.DEBATE_TRANSCRIPT_MARKDOWN, "Transcript not available.")
    print(transcript_md) # Print directly to console for better markdown rendering if possible

    # Optionally, save transcript to a file
    transcript_file_path = os.path.join(PROJECT_ROOT, "debate_transcript.md")
    try:
        with open(transcript_file_path, "w", encoding="utf-8") as f:
            f.write(f"# Debate Transcript\n\n")
            f.write(f"**Topic:** {final_session_state.get(contract.TOPIC)}\n\n")
            f.write(transcript_md)
        logger.info(f"Debate transcript saved to: {transcript_file_path}")
    except Exception as e:
        logger.error(f"Failed to save transcript: {e}")

if __name__ == "__main__":
    main()
