
import re
import os
from pydantic import PrivateAttr
from typing import Dict, Any, Tuple, Optional

from google.adk import Agent
from ..core import llm_clients, session_state_contract as contract
from .base_llm_agent import BaseLLMAgent

class ModeratorAgent(BaseLLMAgent):
    """
    An agent responsible for moderating the debate, summarizing rounds,
    and scoring debaters based on predefined rubrics.
    """
    _prompt_template: str = PrivateAttr()

    def __init__(self, agent_id: str, llm_service_name: str, prompt_file_path: str, **kwargs):
        """
        Initializes the ModeratorAgent.

        Args:
            agent_id (str): The unique identifier for this agent instance.
            llm_service_name (str): The name of the LLM service to use (e.g., "gemini", "openai").
            prompt_file_path (str): The absolute path to the .txt file containing the prompt template.
        """
        super().__init__(agent_id=agent_id, llm_service_name=llm_service_name)
        # self.name is set by BaseLLMAgent's super().__init__(name=agent_id)
        # self.llm_service_name is also set by BaseLLMAgent's super()
        try:
            with open(prompt_file_path, 'r', encoding='utf-8') as f:
                self._prompt_template = f.read()
        except FileNotFoundError:
            print(f"Error: Prompt file not found at {prompt_file_path}")
            self._prompt_template = "Error: Prompt file not found." # Fallback
        except Exception as e:
            print(f"Error loading prompt file {prompt_file_path}: {e}")
            self._prompt_template = f"Error loading prompt: {e}" # Fallback

    def _parse_moderator_response(self, llm_response: str) -> Dict[str, Any]:
        """
        Parses the LLM's response to extract structured information:
        - Round Summary
        - Pro Scores (Argument Quality, Rebuttal Effectiveness, Strategic Positioning) & Justification
        - Con Scores (Argument Quality, Rebuttal Effectiveness, Strategic Positioning) & Justification

        This is a placeholder and will need robust implementation based on the
        expected output format from the LLM.
        """
        parsed_data = {
            "round_summary": "N/A",
            "pro_score_arg_quality": 0.0,
            "pro_score_rebuttal": 0.0,
            "pro_score_strategy": 0.0,
            "pro_justification": "N/A",
            "con_score_arg_quality": 0.0,
            "con_score_rebuttal": 0.0,
            "con_score_strategy": 0.0,
            "con_justification": "N/A",
        }
        # Placeholder parsing logic - this needs to be robust
        # Example: using regex to find sections and scores
        # For now, we'll assume a very basic, structured response for demonstration
        
        # Attempt to find a summary
        summary_match = re.search(r"\*\*Round Summary:\*\*\s*(.*?)\s*(?:\*\*Pro Debater Scores:\*\*|\*\*Con Debater Scores:\*\*|$)", llm_response, re.DOTALL | re.IGNORECASE)
        print("--- PARSING LLM RESPONSE ---")
        if summary_match:
            parsed_data["round_summary"] = summary_match.group(1).strip()
        else:
            # Fallback if '**Round Summary:**' is not found, try without asterisks or a more generic approach
            summary_match = re.search(r"Round Summary:\s*(.*?)\s*(?:Pro Debater Scores:|Con Debater Scores:|$)", llm_response, re.DOTALL | re.IGNORECASE)
            if summary_match:
                parsed_data["round_summary"] = summary_match.group(1).strip()
            else:
                print("Summary NOT matched.")

        # Helper function to extract score, allowing for 'N/A'
        def get_score(score_text):
            if score_text and score_text.lower().strip().startswith('n/a'):
                return 0.0
            try:
                return float(score_text)
            except (ValueError, TypeError):
                return 0.0

        # Pro Scores and Justification
        pro_scores_pattern = r"\*\*Pro Debater Scores:\*\*.*?Argument Quality:\s*(\d+\.?\d*|N/A(?:\s*\(Initial round\))?).*?/10.*?Rebuttal Effectiveness:\s*(\d+\.?\d*|N/A(?:\s*\(Initial round\))?).*?/10.*?Strategic Positioning:\s*(\d+\.?\d*|N/A(?:\s*\(Initial round\))?).*?/10\n\nJustification:\s*(.*?)(?=\n\n\*\*Con Debater Scores:\*\*|\Z)"
        pro_scores_match = re.search(pro_scores_pattern, llm_response, re.DOTALL | re.IGNORECASE)
        if pro_scores_match:
            parsed_data["pro_score_arg_quality"] = get_score(pro_scores_match.group(1))
            parsed_data["pro_score_rebuttal"] = get_score(pro_scores_match.group(2))
            parsed_data["pro_score_strategy"] = get_score(pro_scores_match.group(3))
            parsed_data["pro_justification"] = pro_scores_match.group(4).strip()
        else: # Fallback without asterisks
            pro_scores_pattern_no_ast = r"Pro Debater Scores:.*?Argument Quality:\s*(\d+\.?\d*|N/A(?:\s*\(Initial round\))?).*?/10.*?Rebuttal Effectiveness:\s*(\d+\.?\d*|N/A(?:\s*\(Initial round\))?).*?/10.*?Strategic Positioning:\s*(\d+\.?\d*|N/A(?:\s*\(Initial round\))?).*?/10\n\nJustification:\s*(.*?)(?=\n\nCon Debater Scores:|\Z)"
            pro_scores_match = re.search(pro_scores_pattern_no_ast, llm_response, re.DOTALL | re.IGNORECASE)
            if pro_scores_match:
                parsed_data["pro_score_arg_quality"] = get_score(pro_scores_match.group(1))
                parsed_data["pro_score_rebuttal"] = get_score(pro_scores_match.group(2))
                parsed_data["pro_score_strategy"] = get_score(pro_scores_match.group(3))
                parsed_data["pro_justification"] = pro_scores_match.group(4).strip()
            else:
                print("Pro Scores NOT matched (neither with nor without asterisks).")
        
        # Con Scores and Justification
        con_scores_pattern = r"\*\*Con Debater Scores:\*\*.*?Argument Quality:\s*(\d+\.?\d*|N/A(?:\s*\(Initial round\))?).*?/10.*?Rebuttal Effectiveness:\s*(\d+\.?\d*|N/A(?:\s*\(Initial round\))?).*?/10.*?Strategic Positioning:\s*(\d+\.?\d*|N/A(?:\s*\(Initial round\))?).*?/10.*?Justification:\s*(.*?)\Z"
        con_scores_match = re.search(con_scores_pattern, llm_response, re.DOTALL | re.IGNORECASE)
        if con_scores_match:
            print(f"  Con Groups: {con_scores_match.groups()}")
            parsed_data["con_score_arg_quality"] = get_score(con_scores_match.group(1))
            parsed_data["con_score_rebuttal"] = get_score(con_scores_match.group(2))
            parsed_data["con_score_strategy"] = get_score(con_scores_match.group(3))
            parsed_data["con_justification"] = con_scores_match.group(4).strip()
        else: # Fallback without asterisks
            con_scores_pattern_no_ast = r"Con Debater Scores:.*?Argument Quality:\s*(\d+\.?\d*|N/A(?:\s*\(Initial round\))?).*?/10.*?Rebuttal Effectiveness:\s*(\d+\.?\d*|N/A(?:\s*\(Initial round\))?).*?/10.*?Strategic Positioning:\s*(\d+\.?\d*|N/A(?:\s*\(Initial round\))?).*?/10.*?Justification:\s*(.*?)\Z"
            con_scores_match = re.search(con_scores_pattern_no_ast, llm_response, re.DOTALL | re.IGNORECASE)
            if con_scores_match:
                print(f"  Con Groups: {con_scores_match.groups()}")
                parsed_data["con_score_arg_quality"] = get_score(con_scores_match.group(1))
                parsed_data["con_score_rebuttal"] = get_score(con_scores_match.group(2))
                parsed_data["con_score_strategy"] = get_score(con_scores_match.group(3))
                parsed_data["con_justification"] = con_scores_match.group(4).strip()
            else:
                print("Con Scores NOT matched (neither with nor without asterisks).")
            
        with open("parsed_output_debug.txt", "w", encoding="utf-8") as f_parsed:
            f_parsed.write(str(parsed_data))
        print(f"--- FINAL PARSED DATA from _parse_moderator_response (also saved to parsed_output_debug.txt): ---\n{parsed_data}")
        return parsed_data

    def execute(self, session_state: dict, **kwargs: Any) -> dict:
        """
        Executes the moderator's turn for the current round.

        Retrieves arguments, formats them into the prompt, invokes the LLM,
        parses the response, updates session state, and appends to the transcript.
        """
        print(f"Agent '{self.name}': Executing...")

        debate_topic = session_state.get(contract.TOPIC, "Unknown Topic")
        current_round = session_state.get(contract.CURRENT_ROUND, 0)
        
        pro_args_by_round = session_state.get(contract.PRO_ARGUMENTS_BY_ROUND, {})
        con_args_by_round = session_state.get(contract.CON_ARGUMENTS_BY_ROUND, {})

        pro_argument = pro_args_by_round.get(str(current_round), "No argument presented by Pro.")
        con_argument = con_args_by_round.get(str(current_round), "No argument presented by Con.")

        # Construct the input for the prompt
        # This part will need to align with the placeholders in your moderator_prompt.txt
        # For now, creating a dynamic section to append to the main prompt template
        round_context_prompt = (
            f"\\n\\n--- Current Round Details ---\\n"
            f"Debate Topic: {debate_topic}\\n"
            f"Current Round Number: {current_round}\\n"
            f"Pro Debater's Argument for this Round:\\n{pro_argument}\\n\\n"
            f"Con Debater's Argument for this Round:\\n{con_argument}\\n\\n"
            f"--- Moderator Task for this Round ---\\n"
            f"Based on the provided arguments for Round {current_round} and the overall debate context, please fulfill your per-round responsibilities as outlined in your primary instructions. Specifically:\\n"
            f"1. Provide a concise summary of the key points and arguments presented by each participant in this round.\\n"
            f"2. Identify key disagreements/clashes from this round.\\n"
            f"3. Score this round for BOTH Pro and Con debaters using the 'Per-Round Scoring' rubric (Argument Quality, Rebuttal Effectiveness, Strategic Positioning - each out of 10 points). \\n"
            f"4. Provide brief justifications for the scores awarded to each debater for this round.\\n"
            f"Please structure your response clearly, with distinct sections for the summary, Pro scores/justification, and Con scores/justification. For example:\\n"
            f"Round Summary: [Your summary here]\\n"
            f"Pro Debater Scores:\\nArgument Quality: [Score]/10\\nRebuttal Effectiveness: [Score]/10\\nStrategic Positioning: [Score]/10\\nJustification: [Your justification for Pro's scores]\\n"
            f"Con Debater Scores:\\nArgument Quality: [Score]/10\\nRebuttal Effectiveness: [Score]/10\\nStrategic Positioning: [Score]/10\\nJustification: [Your justification for Con's scores]"
        )
        
        full_prompt = self._prompt_template + round_context_prompt

        print(f"Agent '{self.name}': Invoking LLM for round {current_round} moderation...")
        # print(f"Agent '{self.name}': Full prompt for LLM: \\n{full_prompt[:500]}...") # Print start of prompt for debugging
        
        llm_response_text = self._invoke_llm(prompt=full_prompt, **kwargs)
        
        if not llm_response_text:
            print(f"Agent '{self.name}': LLM invocation failed or returned empty response.")
            # Update session state with error/status
            session_state[contract.LAST_AGENT_STATUS] = f"ModeratorAgent: LLM failed for round {current_round}"
            session_state[contract.ERROR_MESSAGE] = "Moderator LLM response was empty."
            # Append to transcript
            transcript = session_state.get(contract.DEBATE_TRANSCRIPT_MARKDOWN, "")
            transcript += f"\n\n---\n**Moderator - Round {current_round}**\n*LLM invocation failed or returned empty response.*\n---\n"
            session_state[contract.DEBATE_TRANSCRIPT_MARKDOWN] = transcript
            return session_state

        print(f"Agent '{self.name}': LLM response received. Parsing...")
        with open("llm_raw_output.txt", "w", encoding="utf-8") as f_raw:
            f_raw.write(llm_response_text)
        print(f"Agent '{self.name}': Raw LLM response saved to llm_raw_output.txt")
        # print(f"Agent '{self.name}': LLM Raw Response (first 500 chars):\n{llm_response_text[:500]}")

        parsed_scores = self._parse_moderator_response(llm_response_text)

        # Update session state with scores
        round_scores = session_state.get(contract.ROUND_SCORES, {})
        round_scores[str(current_round)] = {
            "summary": parsed_scores["round_summary"],
            "pro": {
                "arg_quality": parsed_scores["pro_score_arg_quality"],
                "rebuttal": parsed_scores["pro_score_rebuttal"],
                "strategy": parsed_scores["pro_score_strategy"],
                "justification": parsed_scores["pro_justification"]
            },
            "con": {
                "arg_quality": parsed_scores["con_score_arg_quality"],
                "rebuttal": parsed_scores["con_score_rebuttal"],
                "strategy": parsed_scores["con_score_strategy"],
                "justification": parsed_scores["con_justification"]
            }
        }
        session_state[contract.ROUND_SCORES] = round_scores
        print(f"Agent '{self.name}': Updated round scores for round {current_round}.")

        # Append to Markdown transcript
        transcript = session_state.get(contract.DEBATE_TRANSCRIPT_MARKDOWN, "")
        transcript += f"\n\n---\n"
        transcript += f"**Moderator - Round {current_round} Summary & Scores**\n\n"
        transcript += f"**Summary:**\n{parsed_scores['round_summary']}\n\n"
        transcript += f"**Pro Debater Scores (Round {current_round}):**\n"
        transcript += f"- Argument Quality: {parsed_scores['pro_score_arg_quality']}/10\n"
        transcript += f"- Rebuttal Effectiveness: {parsed_scores['pro_score_rebuttal']}/10\n"
        transcript += f"- Strategic Positioning: {parsed_scores['pro_score_strategy']}/10\n"
        transcript += f"  - *Justification:* {parsed_scores['pro_justification']}\n\n"
        transcript += f"**Con Debater Scores (Round {current_round}):**\n"
        transcript += f"- Argument Quality: {parsed_scores['con_score_arg_quality']}/10\n"
        transcript += f"- Rebuttal Effectiveness: {parsed_scores['con_score_rebuttal']}/10\n"
        transcript += f"- Strategic Positioning: {parsed_scores['con_score_strategy']}/10\n"
        transcript += f"  - *Justification:* {parsed_scores['con_justification']}\n"
        transcript += f"---\n"
        session_state[contract.DEBATE_TRANSCRIPT_MARKDOWN] = transcript
        
        session_state[contract.LAST_AGENT_STATUS] = f"ModeratorAgent: Successfully moderated round {current_round}"
        print(f"Agent '{self.name}': Successfully moderated round {current_round}.")
        return session_state

# --- Test Block ---
if __name__ == '__main__':
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '..', '.env'))

    # For testing, ensure environment variables for LLM clients are set in .env
    # e.g., GEMINI_API_KEY or OPENAI_API_KEY

    # Define the relative path to the prompts directory
    PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "..", "prompts")
    MODERATOR_PROMPT_FILE = os.path.join(PROMPTS_DIR, "moderator_prompt.txt")

    print("Testing ModeratorAgent...")

    # Initialize the agent
    # Choose your LLM service: "gemini", "openai", "anthropic"
    # Ensure you have the corresponding API key in your .env file
    # For Gemini, also ensure GOOGLE_GENAI_USE_VERTEXAI is set if needed
    # and GOOGLE_CLOUD_PROJECT, GOOGLE_CLOUD_LOCATION for Vertex
    moderator = ModeratorAgent(
        agent_id="moderator_agent_test",
        llm_service_name="gemini", # or "openai", "anthropic"
        prompt_file_path=MODERATOR_PROMPT_FILE
    )
    
    # Test Case 1: Simulate a session state for round 1 moderation
    print("\n--- Test Case 1: Round 1 Moderation ---")
    initial_session_state = {}
    initial_session_state[contract.TOPIC] = "Is remote work beneficial for productivity?"
    initial_session_state[contract.CURRENT_ROUND] = 1
    initial_session_state[contract.PRO_ARGUMENTS_BY_ROUND] = {
        "1": "Pro debater argues that remote work increases focus by reducing office distractions and allows for a flexible schedule, boosting overall productivity."
    }
    initial_session_state[contract.CON_ARGUMENTS_BY_ROUND] = {
        "1": "Con debater counters that remote work leads to communication challenges, potential isolation, and difficulty in maintaining a strong team culture, which can hinder productivity."
    }
    initial_session_state[contract.DEBATE_TRANSCRIPT_MARKDOWN] = "# Debate Topic: Is remote work beneficial for productivity?\n\n"
    initial_session_state[contract.ROUND_SCORES] = {}
    
    # Execute the agent
    # You might need to pass llm_specific_params if your BaseLLMAgent or chosen LLM requires them
    # e.g., model_name for OpenAI: openai_params={"model": "gpt-3.5-turbo"}
    # For Gemini, if using non-Vertex, it defaults to a model.
    # Vertex may require a model name via gemini_params.
    updated_session_state = moderator.execute(
        initial_session_state,
        # Example for OpenAI:
        # openai_params={"model": "gpt-3.5-turbo", "max_tokens": 1000, "temperature": 0.7}
        # Example for Anthropic:
        # anthropic_params={"model": "claude-3-opus-20240229", "max_tokens": 1000}
        # Example for Gemini (Vertex AI specific model):
        # gemini_params={"model_name":"gemini-1.5-flash-001"}
        # Example for Gemini (Google AI Studio - uses default model if not specified):
        gemini_params={} # or specify e.g. {"generation_config": {"temperature": 0.7}}
    )

    print("\\n--- Verification ---")
    print(f"Topic: {updated_session_state.get(contract.TOPIC)}")
    print(f"Current Round: {updated_session_state.get(contract.CURRENT_ROUND)}")
    
    round_1_scores = updated_session_state.get(contract.ROUND_SCORES, {}).get("1", {})
    print(f"Round 1 Scores: {round_1_scores}")
    assert "summary" in round_1_scores
    assert "pro" in round_1_scores and "arg_quality" in round_1_scores["pro"]
    assert "con" in round_1_scores and "arg_quality" in round_1_scores["con"]

    print("\\n--- Final Markdown Transcript (Excerpt) ---")
    final_transcript = updated_session_state.get(contract.DEBATE_TRANSCRIPT_MARKDOWN, "")
    # Print last 500 characters as it can be long
    print(final_transcript[-1000:]) 
    assert "Moderator - Round 1 Summary & Scores" in final_transcript
    assert "Pro Debater Scores (Round 1):" in final_transcript
    assert "Con Debater Scores (Round 1):" in final_transcript

    print("\\nModeratorAgent test completed.")