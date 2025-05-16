import os
from typing import Dict, Any, Optional
# from ..core.session_state_contract import SessionStateKeys # Commented out as it's not used and causes ImportError
from .base_llm_agent import BaseLLMAgent
from pydantic import PrivateAttr

class DebaterAgent(BaseLLMAgent):
    """
    An agent that takes on a stance in a debate, generates arguments,
    and rebuts opposing viewpoints based on a provided prompt template.
    """
    _prompt_template: Optional[str] = PrivateAttr(default=None) # Stores the loaded prompt

    def __init__(self,
                 agent_id: str,
                 llm_service_name: str,
                 prompt_file_path: str,
                 **kwargs):
        super().__init__(agent_id=agent_id, llm_service_name=llm_service_name, **kwargs)
        self._prompt_template = self._load_prompt_template(prompt_file_path)
        if not self._prompt_template:
            # Propagate the error more explicitly if prompt loading fails critical initialization
            raise FileNotFoundError(f"Critical Error: Prompt template could not be loaded from {prompt_file_path}. Agent '{agent_id}' cannot operate.")

    def _load_prompt_template(self, file_path: str) -> Optional[str]:
        """Loads the prompt template from the given file path."""
        try:
            # Ensure the file_path is treated as is. Caller should ensure it's correct.
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            print(f"Error: Prompt file not found at {file_path}")
            return None
        except Exception as e:
            print(f"Error loading prompt file {file_path}: {e}")
            return None

    def execute(self, session_state: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Executes the debater agent's turn.

        Args:
            session_state: The current session state (not directly used by agent logic here but part of ADK signature).
            **kwargs: Expected to contain:
                debate_topic (str): The topic of the debate.
                assigned_stance (Optional[str]): The stance assigned to this agent. 
                                                 If None, the agent forms its own stance as per its prompt.
                debate_history (str): A string representing the history of the debate so far.
                llm_call_kwargs (Optional[Dict]): Additional kwargs for _invoke_llm (e.g. service specific params)

        Returns:
            A dictionary containing the agent's response.
        """
        print(f"Agent {self.name}: Executing turn.")

        debate_topic = kwargs.get("debate_topic")
        assigned_stance_str = kwargs.get("assigned_stance", "[None - Formulate your own]")
        debate_history = kwargs.get("debate_history", "No debate history yet.")
        llm_call_kwargs = kwargs.get("llm_call_kwargs", {})

        if not debate_topic:
            print(f"Error for Agent {self.name}: Debate topic not provided.")
            return {"agent_id": self.name, "error": "Debate topic not provided."}

        input_context_for_llm = (
            f"Debate Topic: {debate_topic}\n"
            f"Your Assigned Stance: {assigned_stance_str}\n"
            f"Debate Context/History:\n{debate_history}\n\n"
            f"---\n\n"
        )
        full_prompt_for_llm = input_context_for_llm + self._prompt_template

        # Ensure default max_tokens for services that might need it for longer debate responses
        if self.llm_service_name == "anthropic":
            anthropic_params = llm_call_kwargs.get('anthropic_params', {})
            anthropic_params.setdefault('max_tokens', 2048)
            llm_call_kwargs['anthropic_params'] = anthropic_params
        elif self.llm_service_name == "openai":
            openai_params = llm_call_kwargs.get('openai_params', {})
            openai_params.setdefault('max_tokens', 2048)
            llm_call_kwargs['openai_params'] = openai_params

        try:
            print(f"Agent {self.name}: Invoking LLM for topic '{debate_topic}'. Assigned stance: '{assigned_stance_str}'.")
            llm_response_text = self._invoke_llm(full_prompt_for_llm, **llm_call_kwargs)
        except Exception as e:
            print(f"Agent {self.name}: Error invoking LLM: {e}")
            return {"agent_id": self.name, "error": f"LLM invocation failed: {e}"}

        current_stance_parsed = None
        if assigned_stance_str == "[None - Formulate your own]":
            lines = llm_response_text.split('\n', 1)
            if lines and lines[0].lower().startswith("[your stance"):
                try:
                    stance_line = lines[0]
                    current_stance_parsed = stance_line[stance_line.find(':')+1:].replace(']', '').strip()
                    print(f"Agent {self.name}: Parsed newly formed stance: '{current_stance_parsed}'")
                except Exception as ex:
                    print(f"Agent {self.name}: Could not parse newly formed stance from response line '{lines[0]}': {ex}")
                    current_stance_parsed = "Error: Stance parsing failed."
            else:
                print(f"Agent {self.name}: Did not find a stance declaration in the first line for auto-formed stance.")
        else:
            current_stance_parsed = kwargs.get("assigned_stance") # Keep the initially assigned stance

        print(f"Agent {self.name}: LLM response received (snippet): {llm_response_text[:300]}...")
        return {
            "agent_id": self.name,
            "argument": llm_response_text,
            "current_stance": current_stance_parsed
        }

if __name__ == '__main__':
    from dotenv import load_dotenv
    # Construct .env path relative to this file's location (src/agents/debater_agent.py)
    # Project root is two levels up from src/agents
    project_root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    dotenv_path = os.path.join(project_root_dir, '.env')
    load_dotenv(dotenv_path=dotenv_path)
    print(f"Loaded .env from: {dotenv_path}")

    # Path to the generic debater prompt
    generic_debater_prompt_path = os.path.join(project_root_dir, "src", "prompts", "pro_debater_prompt.txt")
    
    print(f"Attempting to load prompt from: {generic_debater_prompt_path}")
    if not os.path.exists(generic_debater_prompt_path):
        print(f"CRITICAL ERROR: Prompt file does not exist at {generic_debater_prompt_path}. Tests cannot run.")
    else:
        test_services = ["gemini", "openai", "anthropic"]
        # test_services = ["anthropic"] # For focused testing

        for service in test_services:
            print(f"\n--- Testing DebaterAgent with {service.upper()} ---")
            try:
                debater = DebaterAgent(
                    agent_id=f"debater_{service}",
                    llm_service_name=service,
                    prompt_file_path=generic_debater_prompt_path
                )

                print("\n-- Test Case 1: First turn, no assigned stance --")
                result1 = debater.execute(
                    session_state={},
                    debate_topic="Should AI development be paused for safety reasons?",
                    assigned_stance=None, # Let agent formulate its stance
                    debate_history="This is the very first turn of the debate."
                )
                print(f"Result from {service} (Test Case 1):")
                if "error" in result1:
                    print(f"  Error: {result1['error']}")
                else:
                    print(f"  Agent ID: {result1.get('agent_id')}")
                    print(f"  Formed/Current Stance: {result1.get('current_stance')}")
                    print(f"  Argument Snippet: {result1.get('argument', '')[:300].replace('\n', ' ')}...")

                print("\n-- Test Case 2: Subsequent turn, assigned stance 'PRO' --")
                result2 = debater.execute(
                    session_state={},
                    debate_topic="Should AI development be paused for safety reasons?",
                    assigned_stance="PRO - AI development should continue with careful oversight, not paused.",
                    debate_history=(
                        "Previous turn by Opponent:\n"
                        "AI poses an existential threat and we must halt all development immediately "
                        "until we fully understand the risks. Uncontrolled AI evolution could lead to "
                        "unforeseen catastrophic consequences."
                    )
                )
                print(f"Result from {service} (Test Case 2):")
                if "error" in result2:
                    print(f"  Error: {result2['error']}")
                else:
                    print(f"  Agent ID: {result2.get('agent_id')}")
                    print(f"  Current Stance: {result2.get('current_stance')}")
                    print(f"  Argument Snippet: {result2.get('argument', '')[:300].replace('\n', ' ')}...")
                
            except FileNotFoundError as fnf_error:
                 print(f"  FileNotFoundError during {service} agent initialization: {fnf_error}")        
            except Exception as e:
                print(f"  An unexpected error occurred with {service}: {e}")
                import traceback
                traceback.print_exc()

        print("\nDebaterAgent testing finished.")
