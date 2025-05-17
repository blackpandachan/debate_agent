import os
import re
from typing import Any, Dict, Optional, Tuple

from google.adk.agents import Agent
from .base_llm_agent import BaseLLMAgent # Assuming BaseLLMAgent is in this path

class FinalEvaluatorAgent(BaseLLMAgent):
    """
    Agent that conducts a holistic evaluation of the entire debate,
    determines a winner, and provides reasoning.
    """
    _prompt_template: Optional[str] # For Pydantic compatibility with BaseLLMAgent

    def __init__(self,
                 agent_id: str,
                 llm_service_name: str,
                 prompt_file_path: str,
                 **kwargs):
        # Load prompt template content into a local variable first.
        loaded_prompt_content = self._load_prompt_template(prompt_file_path)
        if not loaded_prompt_content:
            raise FileNotFoundError(f"Critical Error: Prompt template could not be loaded from {prompt_file_path}. Agent '{agent_id}' cannot operate.")

        adk_agent_params = {
            "adk_instruction": loaded_prompt_content,
            "adk_description": f"Final evaluator agent {agent_id} using {llm_service_name}."
        }

        # Determine ADK model name based on llm_service_name
        if llm_service_name.lower() == "gemini":
            adk_agent_params["adk_model_name"] = kwargs.pop("adk_model_name", os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-flash-latest"))
        elif llm_service_name.lower() == "openai":
             adk_agent_params["adk_model_name"] = kwargs.pop("adk_model_name", os.getenv("OPENAI_MODEL_NAME", "gpt-3.5-turbo"))
        elif llm_service_name.lower() == "anthropic":
             adk_agent_params["adk_model_name"] = kwargs.pop("adk_model_name", os.getenv("ANTHROPIC_MODEL_NAME", "claude-3-haiku-20240307"))
        
        adk_agent_params.update(kwargs)

        super().__init__(agent_id=agent_id,
                         llm_service_name=llm_service_name,
                         **adk_agent_params)
        
        self._prompt_template = loaded_prompt_content # Assign after super().__init__()

    def _load_prompt_template(self, file_path: str) -> Optional[str]:
        """Loads the prompt template from the given file path."""
        try:
            resolved_file_path = os.path.realpath(file_path)
            if not os.path.exists(resolved_file_path) or not os.path.isfile(resolved_file_path):
                print(f"Error: Prompt file not found or is not a regular file at {resolved_file_path}")
                return None
            with open(resolved_file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"Error loading prompt file {file_path}: {e}")
            return None

    def _parse_evaluation_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parses the LLM response to extract structured evaluation data.
        This is a basic implementation and might need to be more robust
        based on the actual LLM output format.
        """
        parsed_data = {
            "final_evaluation_pro": {"score": 0.0, "assessment": "Parsing failed or not provided."},
            "final_evaluation_con": {"score": 0.0, "assessment": "Parsing failed or not provided."},
            "winner_determination": "DRAW",
            "final_reasoning": "Parsing failed or not provided."
        }

        try:
            pro_score_match = re.search(r"\*\*Pro Evaluation Score:\*\*\s*([0-9.]+)", response_text, re.IGNORECASE)
            if pro_score_match:
                parsed_data["final_evaluation_pro"]["score"] = float(pro_score_match.group(1))

            pro_assessment_match = re.search(r"\*\*Pro Evaluation Assessment:\*\*\s*(.+?)(?=\*\*Con Evaluation Score:\*\*|\*\*Winner Determination:\*\*|$)", response_text, re.IGNORECASE | re.DOTALL)
            if pro_assessment_match:
                parsed_data["final_evaluation_pro"]["assessment"] = pro_assessment_match.group(1).strip()

            con_score_match = re.search(r"\*\*Con Evaluation Score:\*\*\s*([0-9.]+)", response_text, re.IGNORECASE)
            if con_score_match:
                parsed_data["final_evaluation_con"]["score"] = float(con_score_match.group(1))
            
            con_assessment_match = re.search(r"\*\*Con Evaluation Assessment:\*\*\s*(.+?)(?=\*\*Winner Determination:\*\*|$)", response_text, re.IGNORECASE | re.DOTALL)
            if con_assessment_match:
                parsed_data["final_evaluation_con"]["assessment"] = con_assessment_match.group(1).strip()

            winner_match = re.search(r"\*\*Winner Determination:\*\*\s*(PRO|CON|DRAW)", response_text, re.IGNORECASE)
            if winner_match:
                parsed_data["winner_determination"] = winner_match.group(1).upper()

            reasoning_match = re.search(r"\*\*Final Reasoning:\*\*\s*(.+)", response_text, re.IGNORECASE | re.DOTALL)
            if reasoning_match:
                parsed_data["final_reasoning"] = reasoning_match.group(1).strip()
                
        except Exception as e:
            print(f"Error parsing LLM evaluation response: {e}\nRaw response:\n{response_text}")
            # Keep default values in case of partial parsing failure
            
        return parsed_data

    def execute(self, session_state: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        print(f"Agent {self.name}: Executing final evaluation.")
        
        topic = session_state.get("topic", "N/A")
        max_rounds = session_state.get("max_rounds", 0)
        debate_history_str = "\n".join([f"{item.get('speaker', 'System')}: {item.get('content', '')}" for item in session_state.get("debate_history", [])])
        
        round_scores = session_state.get("round_scores", {})
        round_scores_str = []
        for r, scores in round_scores.items():
            round_scores_str.append(f"Round {r}: Pro Score: {scores.get('pro_score', 'N/A')}, Con Score: {scores.get('con_score', 'N/A')}, Summary: {scores.get('summary', 'N/A')}")
        round_scores_summary = "\n".join(round_scores_str)

        if not self._prompt_template:
            print(f"Error: Agent {self.name} cannot execute without a loaded prompt template.")
            session_state["error_message"] = f"FinalEvaluatorAgent {self.name} missing prompt template."
            return session_state

        prompt_context = {
            "topic": topic,
            "max_rounds": str(max_rounds),
            "full_debate_history": debate_history_str,
            "per_round_scores": round_scores_summary
        }
        
        # The actual invocation of the LLM is handled by the BaseLLMAgent's invoke_llm method
        # which uses the adk_instruction (our _prompt_template) and formats it with context.
        # We expect BaseLLMAgent.invoke_llm to return the raw string response.
        try:
            llm_response_text = self.invoke_llm(prompt_context=prompt_context, **kwargs) # Pass context for formatting
            if llm_response_text:
                parsed_evaluation = self._parse_evaluation_response(llm_response_text)
                session_state.update(parsed_evaluation)
                print(f"Agent {self.name}: Final evaluation completed. Winner: {parsed_evaluation.get('winner_determination')}")
            else:
                print(f"Error: Agent {self.name} received no response from LLM.")
                session_state["error_message"] = f"FinalEvaluatorAgent {self.name} received no LLM response."
                # Set default failure values
                session_state.update(self._parse_evaluation_response("")) # get default failure structure
        except Exception as e:
            print(f"Error during FinalEvaluatorAgent execution: {e}")
            session_state["error_message"] = f"FinalEvaluatorAgent {self.name} failed: {e}"
            session_state.update(self._parse_evaluation_response("")) # get default failure structure
            
        return session_state


class ScoreAggregationAgent(Agent):
    """
    Agent that calculates the final weighted scores based on round scores
    and the final holistic evaluation.
    """
    def __init__(self, name: str = "ScoreAggregationAgent", **kwargs):
        super().__init__(name=name, **kwargs)

    def run(self, session_state: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        print(f"Agent {self.name}: Aggregating scores.")
        
        round_scores_dict = session_state.get("round_scores", {})
        final_eval_pro = session_state.get("final_evaluation_pro", {"score": 0.0})
        final_eval_con = session_state.get("final_evaluation_con", {"score": 0.0})

        num_rounds = len(round_scores_dict)
        avg_pro_round_score = 0.0
        avg_con_round_score = 0.0

        if num_rounds > 0:
            total_pro_score = sum(details.get("pro_score", 0.0) for r, details in round_scores_dict.items())
            total_con_score = sum(details.get("con_score", 0.0) for r, details in round_scores_dict.items())
            avg_pro_round_score = total_pro_score / num_rounds
            avg_con_round_score = total_con_score / num_rounds
        
        # Weights: Average_Per_Round_Score * 0.25, Final_Holistic_Evaluation_Score * 0.75
        # The proposal originally had 0.25 for round and 0.75 for final.
        # It looks like my thought process swapped it earlier, using the proposal's weights here.
        final_weighted_score_pro = (avg_pro_round_score * 0.25) + (final_eval_pro.get("score", 0.0) * 0.75)
        final_weighted_score_con = (avg_con_round_score * 0.25) + (final_eval_con.get("score", 0.0) * 0.75)
        
        session_state["final_weighted_score_pro"] = final_weighted_score_pro
        session_state["final_weighted_score_con"] = final_weighted_score_con
        
        print(f"Agent {self.name}: Score aggregation complete. Pro: {final_weighted_score_pro:.2f}, Con: {final_weighted_score_con:.2f}")
        return session_state