from abc import ABC, abstractmethod
from google.adk.agents import Agent
import os
from pydantic import PrivateAttr
from typing import Any, Dict, Optional, List
from google.generativeai import types as genai_types # Added for GenerationConfig
from core.llm_clients import get_gemini_client, get_openai_client, get_anthropic_client

class BaseLLMAgent(Agent, ABC):
    llm_service_name: str  # Standard Pydantic field declaration
    _llm_client: Any = PrivateAttr(default=None)  # Declare as PrivateAttr
    """
    Abstract base class for agents that interact with Large Language Models.
    Handles LLM client initialization and provides a common interface.
    """
    def __init__(self,
                 agent_id: str,
                 llm_service_name: str,
                 # Params for google.adk.Agent base class
                 adk_model_name: Optional[str] = None,
                 adk_instruction: Optional[str] = None,
                 adk_description: Optional[str] = None,
                 adk_tools: Optional[list] = None,
                 **additional_adk_agent_kwargs # Catches other potential Agent fields
                 ):
        """
        Initializes the BaseLLMAgent.

        Args:
            agent_id (str): The unique identifier for this agent instance.
            llm_service_name (str): The name of the LLM service to use 
                                      (e.g., 'gemini', 'openai', 'anthropic').
            adk_model_name (Optional[str]): Model name for the google.adk.Agent.
            adk_instruction (Optional[str]): Instruction for the google.adk.Agent.
            adk_description (Optional[str]): Description for the google.adk.Agent.
            adk_tools (Optional[list]): List of tools for the google.adk.Agent.
            **additional_adk_agent_kwargs: Additional keyword arguments for google.adk.Agent.
        """
        # Pass all fields expected by BaseLLMAgent (name from Agent, llm_service_name from self)
        # to the Pydantic initialization mechanism via super().
        # Prepare kwargs for Pydantic Agent initialization
        agent_constructor_kwargs = {"name": agent_id} # 'name' is a field of google.adk.Agent
        if adk_model_name:
            agent_constructor_kwargs["model"] = adk_model_name
        if adk_instruction:
            agent_constructor_kwargs["instruction"] = adk_instruction
        if adk_description:
            agent_constructor_kwargs["description"] = adk_description
        if adk_tools:
            agent_constructor_kwargs["tools"] = adk_tools
        
        agent_constructor_kwargs.update(additional_adk_agent_kwargs)

        # Initialize Pydantic models (Agent's fields via **agent_constructor_kwargs, 
        # and BaseLLMAgent's own field 'llm_service_name' explicitly).
        super().__init__(llm_service_name=llm_service_name.lower(), **agent_constructor_kwargs)
        # self.llm_service_name is now set by Pydantic. self.name is also set.
        # self._llm_client is initialized via PrivateAttr(default=None) or by _initialize_llm_client.
        self._initialize_llm_client() # Uses self.llm_service_name

    def _initialize_llm_client(self):
        """Initializes the appropriate LLM client based on llm_service_name."""
        if self.llm_service_name == "gemini":
            gemini_client_or_module = get_gemini_client() # This is called by llm_clients.py
            if gemini_client_or_module:
                use_vertex_ai = os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "False").lower() == "true"
                if use_vertex_ai:
                    # For Vertex AI, get_gemini_client() from llm_clients.py already returns an instantiated model
                    self._llm_client = gemini_client_or_module
                    print(f"Agent '{self.name}': Initialized Gemini client via Vertex AI. Client type: {type(self._llm_client)}")
                else:
                    # For Google AI Studio, get_gemini_client() from llm_clients.py returns the genai module.
                    # We need to instantiate a GenerativeModel from it.
                    try:
                        # Default to a fast and common model if not specified by GEMINI_MODEL_NAME
                        model_name = os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-flash-latest") 
                        # Define a default generation config for the model instance
                        default_model_gen_config = genai_types.GenerationConfig(candidate_count=1)
                        self._llm_client = gemini_client_or_module.GenerativeModel(
                            model_name,
                            generation_config=default_model_gen_config
                        )
                        print(f"Agent '{self.name}': Initialized Gemini client (model: {model_name}) via Google AI Studio with default GenerationConfig. Client type: {type(self._llm_client)}")
                    except AttributeError as ae:
                        # This might happen if gemini_client_or_module is None or not the expected module
                        print(f"Error: Failed to instantiate Gemini GenerativeModel for agent '{self.name}'. AttributeError: {ae}. Client/Module was: {gemini_client_or_module}")
                        self._llm_client = None
                    except Exception as e:
                        print(f"Error: Unexpected issue instantiating Gemini GenerativeModel for agent '{self.name}': {e}")
                        self._llm_client = None
            else:
                # get_gemini_client() returned None (e.g., API key missing for initial setup of genai module)
                print(f"Agent '{self.name}': Gemini client setup failed (get_gemini_client returned None).")
                self._llm_client = None
        elif self.llm_service_name == "openai":
            self._llm_client = get_openai_client()
        elif self.llm_service_name == "anthropic":
            self._llm_client = get_anthropic_client()
        else:
            raise ValueError(f"Unsupported LLM service: {self.llm_service_name}")

        if not self._llm_client:
            print(f"Warning: LLM client for '{self.llm_service_name}' could not be initialized (likely missing API key or config).")

    def _invoke_llm(self, prompt: str, **kwargs) -> str:
        """
        Invokes the configured LLM with the given prompt.
        kwargs are used to pass service-specific parameters, e.g., kwargs={'openai_params': {'model': 'gpt-4'}}
        """
        if not self._llm_client:
            print(f"Error: LLM client for agent '{self.name}' (service: {self.llm_service_name}) not initialized properly.")
            raise RuntimeError(f"LLM client for {self.llm_service_name} not initialized for agent {self.name}.")

        print(f"Agent '{self.name}': Invoking LLM '{self.llm_service_name}' with prompt: '{prompt[:70]}...' and kwargs: {kwargs}")

        try:
            if self.llm_service_name == "gemini":
                gemini_params = kwargs.get('gemini_params', {})
                if not hasattr(self._llm_client, 'generate_content'):
                    print(f"Error: Gemini client for agent '{self.name}' is not a valid model instance. Client type: {type(self._llm_client)}")
                    return "Error: Invalid Gemini client configuration."

                response = self._llm_client.generate_content(prompt, **gemini_params)
                
                # Robust Gemini response parsing
                if hasattr(response, 'candidates') and response.candidates and \
                   hasattr(response.candidates[0], 'content') and \
                   hasattr(response.candidates[0].content, 'parts') and \
                   response.candidates[0].content.parts:
                    return response.candidates[0].content.parts[0].text
                elif hasattr(response, 'text') and response.text and response.text.strip():
                    return response.text
                elif hasattr(response, 'parts') and response.parts:
                    return response.parts[0].text
                else:
                    block_reason_msg = ""
                    if hasattr(response, 'prompt_feedback') and \
                       hasattr(response.prompt_feedback, 'block_reason') and \
                       response.prompt_feedback.block_reason:
                        block_reason_msg = f" Reason: {response.prompt_feedback.block_reason_message or response.prompt_feedback.block_reason}"
                    print(f"Warning: Gemini response for agent '{self.name}' (service: {self.llm_service_name}) had an unexpected structure or was empty/blocked.{block_reason_msg} Full response: {response}")
                    return f"Error: Could not parse Gemini response or content was empty/blocked.{block_reason_msg}"

            elif self.llm_service_name == "openai":
                openai_params = kwargs.get('openai_params', {})
                final_openai_params = {
                    "model": "gpt-3.5-turbo",
                    "messages": [{"role": "user", "content": prompt}],
                    **openai_params
                }
                response = self._llm_client.chat.completions.create(**final_openai_params)
                return response.choices[0].message.content

            elif self.llm_service_name == "anthropic":
                anthropic_params = kwargs.get('anthropic_params', {})
                final_anthropic_params = {
                    "model": "claude-3-haiku-20240307",
                    "max_tokens": 1024, # Anthropic requires max_tokens
                    "messages": [{"role": "user", "content": prompt}],
                    **anthropic_params
                }
                # Ensure max_tokens is present, as it's required by Anthropic
                if 'max_tokens' not in final_anthropic_params:
                    final_anthropic_params['max_tokens'] = 1024 
                response = self._llm_client.messages.create(**final_anthropic_params)
                return response.content[0].text
            
            else:
                raise RuntimeError(f"LLM invocation logic not implemented for {self.llm_service_name}")

        except Exception as e:
            print(f"Error invoking LLM ({self.llm_service_name}) for agent '{self.name}': {e}")
            # import traceback # Uncomment for detailed debugging
            # traceback.print_exc()
            return f"Error: LLM call failed for {self.llm_service_name}. Details: {str(e)}"


    @abstractmethod
    def execute(self, session_state: dict, **kwargs) -> dict:
        """
        The main execution logic for the agent.
        Subclasses must implement this method.

        Args:
            session_state (dict): The current state of the session/pipeline.
            **kwargs: Additional arguments specific to the agent's execution.

        Returns:
            dict: Updates to be merged into the session_state.
        """
        pass

if __name__ == '__main__':
    # Example usage (for testing purposes)
    print("Testing BaseLLMAgent...")

    class MockLLMAgent(BaseLLMAgent):
        def __init__(self, agent_id: str, llm_service_name: str):
            super().__init__(agent_id=agent_id, llm_service_name=llm_service_name)

        def execute(self, session_state: dict, **kwargs) -> dict:
            print(f"Agent {self.name}: Executing with session_state: {session_state}") # Use self.name
            try:
                prompt = kwargs.get("prompt", "Hello, world!")
                
                invoke_kwargs: Dict[str, Any] = {}
                # Common parameter like max_tokens can be a top-level kwarg for _invoke_llm
                # or nested within service_params. For this test, let's try nesting.

                if self.llm_service_name == "gemini":
                    model_name = kwargs.get("model_name", "gemini-1.5-flash-latest") # Model for Gemini client setup
                    # Model name is configured on the client by _initialize_llm_client
                    # For the test, we'll rely on the default GenerationConfig set during initialization.
                    # If we needed to override, we'd pass: {'generation_config': genai_types.GenerationConfig(...)}
                    invoke_kwargs['gemini_params'] = kwargs.get('gemini_params', {}) # Pass empty dict to use model's default config
                    print(f"Agent {self.name}: Using Gemini model (configured in client): {model_name}, will use default generation_config.")
                elif self.llm_service_name == "openai":
                    model_name = kwargs.get("model_name", "gpt-3.5-turbo")
                    invoke_kwargs['openai_params'] = {'model': model_name, 'max_tokens': 50, **kwargs.get('openai_params', {})}
                    print(f"Agent {self.name}: Using OpenAI model: {model_name}")
                elif self.llm_service_name == "anthropic":
                    model_name = kwargs.get("model_name", "claude-3-haiku-20240307")
                    # Anthropic requires max_tokens
                    anthropic_specific_params = {'model': model_name, 'max_tokens': 50, **kwargs.get('anthropic_params', {})}
                    if 'max_tokens' not in anthropic_specific_params:
                         anthropic_specific_params['max_tokens'] = 50 # Ensure it's there
                    invoke_kwargs['anthropic_params'] = anthropic_specific_params
                    print(f"Agent {self.name}: Using Anthropic model: {model_name}")
                else:
                    print(f"Agent {self.name}: Unknown LLM service '{self.llm_service_name}' for model param setup.")
                    # For unknown services, pass no specific params, relying on _invoke_llm defaults if any

                response = self._invoke_llm(prompt, **invoke_kwargs)
                print(f"Agent {self.name}: LLM Response: {response}") # Use self.name
                return {"llm_response": response}
            except Exception as e:
                print(f"Agent {self.name}: Error during execution: {e}") # Use self.name
                # import traceback # For debugging
                # traceback.print_exc() # For debugging
                return {"error": str(e)}

    # -- Test Gemini --
    try:
        print("\n--- Testing Gemini Agent ---")
        gemini_agent = MockLLMAgent(agent_id="test_gemini_agent", llm_service_name="gemini")
        # Note: GEMINI_API_KEY must be set in .env
        # And if GOOGLE_GENAI_USE_VERTEXAI=True, GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION also needed.
        gemini_agent.execute({}, prompt="What is the speed of light?", model_name='gemini-1.5-flash-latest') 
    except ValueError as ve:
        print(f"ValueError during Gemini agent test: {ve}")
    except Exception as e:
        print(f"General error during Gemini agent test: {e}")

    # -- Test OpenAI --
    try:
        print("\n--- Testing OpenAI Agent ---")
        openai_agent = MockLLMAgent(agent_id="test_openai_agent", llm_service_name="openai")
        # Note: OPENAI_API_KEY must be set in .env
        openai_agent.execute({}, prompt="What is the capital of France?", model_name='gpt-3.5-turbo')
    except ValueError as ve:
        print(f"ValueError during OpenAI agent test: {ve}")
    except Exception as e:
        print(f"General error during OpenAI agent test: {e}")

    # -- Test Anthropic --
    try:
        print("\n--- Testing Anthropic Agent ---")
        anthropic_agent = MockLLMAgent(agent_id="test_anthropic_agent", llm_service_name="anthropic")
        # Note: ANTHROPIC_API_KEY must be set in .env
        anthropic_agent.execute({}, prompt="Why is the sky blue?", model_name='claude-3-haiku-20240307')
    except ValueError as ve:
        print(f"ValueError during Anthropic agent test: {ve}")
    except Exception as e:
        print(f"General error during Anthropic agent test: {e}")

    print("\nBaseLLMAgent testing finished.")
