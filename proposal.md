1. IntroductionThis document outlines the project plan for developing a multi-agent debate engine using the Google Agent Development Kit (ADK). It serves as a comprehensive guideline for implementation, emphasizing adherence to ADK's supported patterns, primitives, and best practices for state management, inter-agent communication (context passing and event-like signaling), tool usage, rate limiting, and multi-LLM provider support. The primary goal is to create a modular, robust, extensible, and well-behaved (re: API limits and ADK patterns) debate system.This plan is specifically structured to guide an AI agent (referred to as "Windsurf") in the development process, providing explicit instructions and highlighting areas where strict adherence to ADK documentation is crucial.2. Core ADK Concepts & PrinciplesTo build this system effectively, "Windsurf" must leverage the following core ADK concepts. For detailed API specifications, method signatures, and usage examples, always refer to the official Google ADK documentation for the specific version you are using.Agents (Agent, LLMAgent):The fundamental building blocks. Each agent encapsulates specific logic and should be designed as a self-contained unit of work.Agent: Base class for all agents. Its primary method is run(session_state: SessionState, **kwargs) -> SessionState (or similar, check ADK docs). The agent receives the current state, performs its actions, and returns the (potentially modified) state.LLMAgent: A specialized agent for interacting with Large Language Models. It should abstract LLM provider specifics and manage prompt templating, LLM calls, and response parsing. Context for the LLM is typically derived from SessionState.Orchestration Primitives (SequentialAgent, ParallelAgent, BranchAgent):These are the primary mechanisms for defining the flow of execution and how agents are composed.SequentialAgent: Executes agents in a defined order. The SessionState returned by one agent is passed to the next.ParallelAgent: Executes agents concurrently. Windsurf must understand how SessionState is handled in parallel execution: Does each parallel agent receive a copy of the state, and how are their outputs merged or made available? This is critical for avoiding race conditions or lost updates. Consult ADK docs for patterns like fan-out/fan-in with SessionState. (See also Section 10.1)BranchAgent: Routes execution to different agents based on conditions evaluated against SessionState. This is key for dynamic flows.State and Context Management (SessionState):Primary Mechanism: SessionState is the central nervous system for your ADK application. It's a dictionary-like object that is passed from one agent to the next (via orchestrators).Context Passing: Agents receive all their necessary input data and context by reading predefined keys from SessionState at the beginning of their run method.Result Propagation: Agents make their results, status updates, or any data intended for subsequent agents available by writing to predefined keys in SessionState before their run method returns.Implicit State: Avoid relying on implicit state within agents that persists across multiple invocations within different pipeline runs. Agents should ideally be stateless regarding the overall pipeline's progression, deriving their operational context from the SessionState they receive for the current run.Event Handling and Callbacks (ADK Perspective):State-Driven Events: ADK does not typically feature a direct, arbitrary inter-agent event bus or callback registration system like some other frameworks. Instead, "events" or "signals" are generally represented by changes in SessionState.An agent can signal an "event" (e.g., "StanceFormulated", "ArgumentGenerated", "ErrorOccurred") by writing a specific status or data to a designated key in SessionState.Subsequent agents, or BranchAgent conditions, can then react to these state changes.Callbacks within an Agent (Async Operations): Callbacks might be used internally within an agent when dealing with asynchronous operations from an SDK (e.g., an LLM client SDK's async call might support callbacks for success/failure). However, the agent's interaction with the ADK framework (receiving SessionState and returning it) remains its primary contract. These internal callbacks should ultimately resolve to update the SessionState that the agent will return.No Direct Inter-Agent Callbacks: Agent A does not directly register a callback function with Agent B. Orchestrators manage the flow.Tools (Tool):Reusable components providing specific functionalities.Context for Tools: Tools receive necessary context (e.g., a search query) as parameters passed by the agent invoking them. The agent sources this context from SessionState.Tool Results: Tool execution results (data or errors) are returned to the calling agent, which then decides how to process these results and update SessionState.3. Project Overview & GoalsThe project aims to create a multi-agent system that can conduct a structured debate on a given topic.Key Goals:Initialization: Set up the debate topic, number of rounds, and initial state.Stance Formation: Allow debater agents to use specified LLMs (e.g., Gemini for Pro, OpenAI for Con, with flexibility for others like Anthropic) to formulate initial stances.Conditional Debate: Proceed only if initial stances differ.Structured Debate Rounds: Agents argue in turns.Moderation: A Moderator agent summarizes and scores each round.Final Evaluation: A Final Evaluator agent assesses the debate.Score Aggregation: Calculate a final weighted score.Robust Rate Limiting & API Quota Management: Implement effective strategies (delays, retries, concurrency control) to prevent hitting API rate limits for all integrated LLM providers.Multi-LLM Provider Support: Design for easy integration and selection of different LLM providers (Gemini, OpenAI, Anthropic, etc.) for various agent roles, managed through configuration.ADK Compliance: Strictly use ADK primitives and patterns, especially for state, context, and event management.4. System Architecture & High-Level FlowThe system will be orchestrated as a pipeline of ADK agents. The flow inherently relies on SessionState being passed through each step.(Conceptual Flow Diagram)+---------------------+     +---------------------------+     +-----------------------+
| Initializer Agent   | --> | ParallelAgent             | --> | BranchAgent           |
| (Set up SessionState)|     | (Pro & Con Stance Form.)  |     | (Check Stance Diff.)  |
+---------------------+     +---------------------------+     +-----------------------+
                                                                  |
                                                                  | (Stances Differ)
                                                                  V
+-----------------------------------------------------------------+
| SequentialAgent (Main Debate Pipeline - Repeats for N Rounds)   |
|   +-----------------------------------------------------------+ |
|   | SequentialAgent (Single Round)                            | |
|   |   1. Pro Debater Agent (Presents Argument/Rebuttal)       | |
|   |   2. Con Debater Agent (Presents Argument/Rebuttal)       | |
|   |   3. Moderator Agent (Summarizes Round, Scores Round)     | |
|   +-----------------------------------------------------------+ |
+-----------------------------------------------------------------+
    |
    V
+---------------------+     +-------------------------+
| Final Evaluator     | --> | Score Aggregation Agent |
| Agent (Overall Eval)|     | (Calculate Final Score) |
+---------------------+     +-------------------------+
Step-by-Step Flow:InitializerAgent:Receives the debate topic and max_rounds.Initializes SessionState with topic, max_rounds, current_round = 0, debate_history = [], round_scores = {}, pro_stance = None, con_stance = None, etc.ParallelAgent (Stance Formation):Contains two agents: ProStanceAgent (e.g., Gemini) and ConStanceAgent (e.g., OpenAI).Both agents run concurrently.Each agent researches (if tools are provided) and formulates an initial stance on the topic.They update SessionState with pro_stance and con_stance respectively.BranchAgent (Conditional Debate Start):Condition Logic: Reads pro_stance and con_stance from SessionState. The condition is true if pro_stance and con_stance are meaningfully different.if_true Branch: Points to the MainDebatePipelineAgent (a SequentialAgent).if_false Branch: Points to an EndDebateEarlyAgent.SequentialAgent (Main Debate Pipeline):This agent iterates max_rounds times. In each iteration, it executes a SingleRoundAgent.It manages current_round incrementing in SessionState.SequentialAgent (Single Round):ProDebaterAgent: Reads SessionState. Uses search tool. Writes its argument to SessionState. Updates debate_history.ConDebaterAgent: Reads SessionState. Uses search tool. Writes its argument to SessionState. Updates debate_history.ModeratorAgent: Reads arguments from SessionState. Summarizes the round, provides per-round scores. Updates round_scores and debate_history in SessionState.FinalEvaluatorAgent:Runs after all rounds are completed.Reads debate_history and round_scores from SessionState.Performs a holistic evaluation.Writes final_evaluation_scores, winner_determination, and final_reasoning to SessionState.ScoreAggregationAgent:Reads round_scores and final_evaluation_scores from SessionState.Calculates the final weighted score.Writes final_weighted_score_pro and final_weighted_score_con to SessionState.5. Detailed Agent DefinitionsEach LLMAgent and Agent will interact with SessionState as its primary means of receiving context and publishing results. API keys for each LLM service must be loaded securely from environment variables.General Agent run Method Structure (Conceptual):# Conceptual - Adhere to actual ADK Agent method signature
class MySpecificAgent(Agent): # Or LLMAgent
    def run(self, session_state: SessionState, **kwargs) -> SessionState:
        # 1. Receive Context: Read necessary data from session_state
        topic = session_state.get("topic")
        current_round = session_state.get("current_round")
        # ... other necessary context variables ...

        # 2. Perform Agent Logic (e.g., call LLM, use tools, calculations)
        # If using tools:
        # tool_input = self._prepare_tool_input(topic, ...) # Prepare from context
        # tool_result = self.my_tool.execute(tool_input) # Or however tools are called
        # self._process_tool_result(tool_result, session_state) # Update state based on tool output

        # If calling LLM (for LLMAgent):
        # llm_response = self.llm_client.call(prompt, context_variables) # Context passed to LLM
        # parsed_output = self._parse_llm_response(llm_response)

        # 3. Publish Results/Update State: Write outputs to session_state
        # session_state["my_agent_output"] = processed_result
        # session_state["my_agent_status"] = "completed" # Example of signaling

        # 4. Return updated session_state
        return session_state
Rate Limiting Strategy for LLMAgents:All LLM calls within any agent should be wrapped with robust rate limiting and error handling logic. This includes:Client-Side Delays: An asyncio.sleep(5) (or configurable duration) before each LLM API call.Retry Mechanism: Implement an automatic retry mechanism with exponential backoff and jitter for transient errors and rate limit errors (e.g., HTTP 429, 5xx).Specific Error Code Handling: Catch and interpret specific HTTP error codes from LLM providers.a. InitializerAgent* Role: Sets up the initial debate environment.* ADK Agent Type: Agent (custom Python logic).* LLM Config: N/A.* Tools: N/A.* SessionState Interaction:* Writes: topic, max_rounds, current_round = 0, debate_history = [], round_scores = {}, pro_stance = None, con_stance = None, pro_arguments_by_round = {}, con_arguments_by_round = {}.* Key Parameters: topic (str), max_rounds (int).b. ProStanceAgent (Part of ParallelAgent)* Role: Formulates the initial "Pro" stance.* ADK Agent Type: LLMAgent.* LLM Config: Configurable (e.g., Gemini). An instance of the appropriate LLM client (or configuration for it) should be passed during agent initialization.* Tools: Search Tool (optional, for initial stance grounding).* SessionState Interaction:* Reads (Context): session_state.get("topic").* Writes (Result/Event): session_state["pro_stance"] = "The formulated stance...", session_state["pro_stance_status"] = "formulated".* Prompt: A specific prompt to generate an initial affirmative stance on the {{topic}}.c. ConStanceAgent (Part of ParallelAgent)* Role: Formulates the initial "Con" stance.* ADK Agent Type: LLMAgent.* LLM Config: Configurable (e.g., OpenAI).* Tools: Search Tool (optional).* SessionState Interaction:* Reads (Context): session_state.get("topic").* Writes (Result/Event): session_state["con_stance"] = "The formulated stance...", session_state["con_stance_status"] = "formulated".* Prompt: A specific prompt to generate an initial opposing stance on the {{topic}}.d. ProDebaterAgent (Part of SingleRoundAgent)* Role: Argues for the affirmative stance in a given round.* ADK Agent Type: LLMAgent.* LLM Config: Configurable (e.g., Gemini, or could be Anthropic).* Tools: Search Tool (mandatory for evidence).* SessionState Interaction:* Reads: topic, current_round, max_rounds, pro_stance, debate_history, con_arguments_by_round (to get opponent's last argument).* Writes: pro_arguments_by_round[current_round] (str), updates debate_history (appends its argument).* Prompt: User-provided "Debate Agent Prompt (Affirmative)" (see section 6), parameterized.e. ConDebaterAgent (Part of SingleRoundAgent)* Role: Argues for the opposing stance in a given round.* ADK Agent Type: LLMAgent.* LLM Config: Configurable (e.g., OpenAI, or could be Anthropic).* Tools: Search Tool (mandatory for evidence).* SessionState Interaction:* Reads: topic, current_round, max_rounds, con_stance, debate_history, pro_arguments_by_round (to get opponent's last argument).* Writes: con_arguments_by_round[current_round] (str), updates debate_history (appends its argument).* Prompt: User-provided "Debate Agent Prompt (Con)" (see section 6), parameterized.f. ModeratorAgent (Part of SingleRoundAgent)* Role: Summarizes the current round's arguments and provides scores for each participant for that round.* ADK Agent Type: LLMAgent.* LLM Config: Configurable (e.g., Gemini).* Tools: N/A (relies on arguments from SessionState).* SessionState Interaction:* Reads: topic, current_round, max_rounds, pro_arguments_by_round[current_round], con_arguments_by_round[current_round], debate_history (up to previous round).* Writes: round_scores[current_round] = {'pro': score, 'con': score, 'justification_pro': str, 'justification_con': str, 'summary': str}, updates debate_history (appends round summary).* Prompt: User-provided "Moderator Prompt (Per Round)" (see section 6), parameterized.g. FinalEvaluatorAgent* Role: Conducts a holistic evaluation of the entire debate and determines a winner.* ADK Agent Type: LLMAgent.* LLM Config: Configurable (e.g., OpenAI or a powerful Gemini/Anthropic model).* Tools: N/A.* SessionState Interaction:* Reads: topic, max_rounds, debate_history (complete), round_scores (all rounds).* Writes: final_evaluation_pro = {'score': int, 'assessment': str}, final_evaluation_con = {'score': int, 'assessment': str}, winner_determination (str: "PRO", "CON", or "DRAW"), final_reasoning (str).* Prompt: User-provided "Final Evaluator Prompt" (see section 6), parameterized.h. ScoreAggregationAgent* Role: Calculates the final weighted scores.* ADK Agent Type: Agent (custom Python logic).* LLM Config: N/A.* Tools: N/A.* SessionState Interaction:* Reads: round_scores, final_evaluation_pro['score'], final_evaluation_con['score'].* Calculates: Average per-round scores for Pro and Con.* Calculates: Final weighted scores using the formula: (Average_Per_Round_Score * 0.25) + (Final_Holistic_Evaluation_Score * 0.75).* Writes: final_weighted_score_pro (float), final_weighted_score_con (float).i. EndDebateEarlyAgent* Role: Concludes the debate if initial stances are not different.* ADK Agent Type: Agent (custom Python logic).* LLM Config: N/A.* Tools: N/A.* SessionState Interaction:* Reads: pro_stance, con_stance.* Writes: A status message to SessionState or logs it (e.g., debate_outcome = "No contest: Stances were too similar.").6. Prompt EngineeringThe prompts provided by the user (in earlier discussions) are the basis. "Windsurf" must ensure they are correctly integrated:Parameterization: All placeholders like {{topic}}, {{current_round}}, {{max_rounds}}, {{your_stance}}, {{debate_history}}, {{opponent_previous_argument}}, etc., must be dynamically filled using values from SessionState before making the LLM call. ADK's LLMAgent typically supports this via prompt templating.Clarity and Role Definition: The prompts clearly define roles and objectives.Input from SessionState (Context for Prompts):For Debate Agent Prompt: {{topic}}, {{current_round}}, {{max_rounds}}, {{your_stance}} (from pro_stance or con_stance in SessionState), {{debate_history}} (relevant parts from SessionState), {{opponent_previous_argument}} (fetched from pro_arguments_by_round or con_arguments_by_round in SessionState for the previous round).For Moderator Prompt: {{topic}}, {{current_round}}, {{max_rounds}}, {{pro_argument_current_round}} (from SessionState), {{con_argument_current_round}} (from SessionState), {{debate_history_previous_rounds}} (from SessionState).For Final Evaluator Prompt: {{topic}}, {{max_rounds}}, {{full_debate_history}} (from SessionState), {{per_round_scores}} (from SessionState).Conciseness for LLM Output: Prompts specify desired output length (e.g., 300-500 words).Structured Output: Prompts request specific output formats. LLMAgent might need custom parsing logic if the LLM doesn't perfectly adhere, or prompts can be refined to request JSON output for easier parsing.Model-Specific Tuning: Prompt effectiveness can vary between LLM providers and models (Gemini, OpenAI, Anthropic). Windsurf should anticipate potential model-specific prompt tuning during testing.7. SessionState SpecificationSessionState is critical. This section is the contract for state and context. Windsurf must treat this section as the definitive "contract" for what data (context) agents can expect and what data (results, status "events") they are responsible for producing in SessionState.topic (str): The main subject of the debate. (Set by InitializerAgent)max_rounds (int): Total number of debate rounds. (Set by InitializerAgent)current_round (int): The current round number, 0-indexed or 1-indexed. (Managed by InitializerAgent and MainDebatePipeline)pro_stance (str): The initial affirmative stance. (Set by ProStanceAgent)pro_stance_status (str): Status of pro stance formulation, e.g., "formulated", "error". (Set by ProStanceAgent)con_stance (str): The initial opposing stance. (Set by ConStanceAgent)con_stance_status (str): Status of con stance formulation. (Set by ConStanceAgent)debate_history (list of dicts/str): A chronological record of arguments and summaries. Each entry could be {'round': N, 'speaker': 'Pro/Con/Moderator', 'content': '...'}. (Appended by DebaterAgents, ModeratorAgent)pro_arguments_by_round (dict): {round_num: argument_str} for the Pro debater. (Set by ProDebaterAgent)con_arguments_by_round (dict): {round_num: argument_str} for the Con debater. (Set by ConDebaterAgent)round_scores (dict):  {round_num: {'pro_score': float, 'con_score': float, 'summary': str, 'pro_justification': str, 'con_justification': str}}. (Set by ModeratorAgent)final_evaluation_pro (dict): {'score': float, 'assessment': str}. (Set by FinalEvaluatorAgent)final_evaluation_con (dict): {'score': float, 'assessment': str}. (Set by FinalEvaluatorAgent)winner_determination (str): "PRO", "CON", or "DRAW". (Set by FinalEvaluatorAgent)final_reasoning (str): Justification for the winner. (Set by FinalEvaluatorAgent)final_weighted_score_pro (float): Final calculated score for Pro. (Set by ScoreAggregationAgent)final_weighted_score_con (float): Final calculated score for Con. (Set by ScoreAggregationAgent)debate_outcome (str): Overall status, e.g., "Completed", "No contest: Stances were too similar.", "Error: Max retries reached". (Set by various agents upon conclusion or error)error_message (str, optional): Detailed error message if an agent encounters a critical failure.last_agent_status (dict, optional): {agent_name: "completed/failed", "timestamp": ...} for debugging.8. Tool Integration StrategyTools are invoked by agents. The context for tool execution comes from the agent, which sources it from SessionState.Define the Tool:Create a class that inherits from ADK's Tool base class (Windsurf: verify exact base class and methods from ADK docs).Implement the method responsible for tool execution (e.g., _run(**kwargs) or execute(**kwargs)). This method will perform the actual search (e.g., using a Google Search API). It should accept necessary parameters like a query string.Example:# from adk.tools import Tool # Fictional import, Windsurf: check actual ADK import

class WebSearchTool(Tool): # Windsurf: Ensure this inherits correctly
    name = "WebSearch" # Standardized name for LLM to call
    description = "Performs a web search for a given query and returns a summary of top results." # For LLM to understand tool

    def __init__(self, api_key: str, max_results: int = 3): # Or however API keys/config are managed
        super().__init__()
        self.api_key = api_key
        self.max_results = max_results
        # Initialize search client (e.g., Google Custom Search API client)
        print(f"WebSearchTool initialized. Max results: {self.max_results}")

    # Windsurf: Verify method signature (e.g., _run, execute) and parameters from ADK docs
    def execute(self, query: str) -> str: # Or dict/list of results
        print(f"WebSearchTool executing query: {query}")
        # ... perform search using self.api_key and self.max_results ...
        # ... handle errors, format results (e.g., concatenate snippets) ...
        # Example placeholder result:
        results_summary = f"Search results for '{query}': Result 1 snippet... Result 2 snippet..."
        if not results_summary: # Handle no results found
            return f"No search results found for query: {query}"
        return results_summary
Instantiate and Pass Tools to Agents:When creating instances of ProDebaterAgent and ConDebaterAgent (or any agent needing tools), pass an instance of WebSearchTool (or other tools) to their constructor.ADK LLMAgents typically have a mechanism to be initialized with a list of tools. The agent's prompt usually needs to be aware of available tools (by name and description) so the LLM knows it can request their use.Reference: "Windsurf: Consult the ADK documentation on how LLMAgents are equipped with and invoke tools. The agent's prompt must list available tools and their descriptions."Agent Invoking Tools (Context Flow):An agent's run method reads necessary data from SessionState.This data is used to formulate the inputs/parameters for the tool (e.g., constructing a search query).The agent (or the ADK framework on behalf of the LLM within an LLMAgent) calls the tool (e.g., search_results = self.search_tool.execute(query=derived_query)).The tool returns its result (or raises an error) to the agent.The agent processes the tool's result and updates SessionState accordingly (e.g., session_state["search_findings_for_round_X"] = search_results).Callbacks for Tools: Generally, tool execution within an agent will be synchronous (result = tool.execute()) or asynchronous (result = await tool.execute_async()) from the agent's perspective. If a tool itself uses internal callbacks for its own async operations, these should be managed within the tool and resolve to a final return value or exception for the agent. The agent does not typically pass callbacks into the tool in an ADK context for inter-agent signaling.9. Implementation StepsEnvironment Setup:Install the correct version of Google ADK.Set up environment variables for API keys (Gemini, OpenAI, Anthropic, Search, etc.) in a .env file. Ensure distinct keys are used for each service. Use python-dotenv to load them.Define SessionState Contract EXPLICITLY:Before implementing agents, meticulously list all keys expected in SessionState as detailed in Section 7. For each key, define:Its purpose (what context it provides, what result/event it signifies).Which agent(s) produce it.Which agent(s) consume it.This serves as the central agreement for inter-agent data flow.LLM Client Configuration/Wrapper (If needed):"Windsurf" must investigate how ADK's LLMAgent ingests LLM configurations for different providers (Gemini, OpenAI, Anthropic).If ADK doesn't provide a sufficiently unified interface, a thin wrapper or factory pattern might be needed to provide a consistent way to instantiate and call these clients.This layer must incorporate robust rate limiting logic:Client-Side Delays: asyncio.sleep(CONFIGURED_DELAY) (e.g., 5 seconds) before each LLM call.Retry Mechanism: Automatic retries with exponential backoff and jitter for transient errors (e.g., HTTP 429, 5xx).Configuration: Allow delay times and retry counts to be configurable.Example structure for a hypothetical LLM client wrapper:# Hypothetical example - ADK may have its own way
import asyncio
import time
import random # For jitter
from abc import ABC, abstractmethod

class BaseLLMClient(ABC):
    def __init__(self, api_key, rate_limit_delay=5, max_retries=3):
        self.api_key = api_key
        self.rate_limit_delay = rate_limit_delay
        self.max_retries = max_retries
        # Initialize actual client SDK (e.g., OpenAI, Gemini SDK) here

    async def call_llm(self, prompt_data: dict) -> str: # prompt_data could be a dict with prompt, model, etc.
        retries = 0
        last_exception = None
        while retries <= self.max_retries:
            try:
                print(f"Waiting for {self.rate_limit_delay}s before LLM call (attempt {retries+1})...")
                await asyncio.sleep(self.rate_limit_delay)
                response = await self._execute_llm_call(prompt_data)
                return response # Success
            except Exception as e: # Windsurf: Catch specific API exceptions from SDKs
                last_exception = e
                print(f"LLM call failed (attempt {retries+1}/{self.max_retries+1}): {e}")
                if not self._is_retryable_error(e) or retries == self.max_retries:
                    print(f"Non-retryable error or max retries reached for LLM call.")
                    raise # Non-retryable or max retries reached
                backoff_time = (2 ** retries) + (random.uniform(0, 1)) # Exponential backoff with jitter
                print(f"Retrying LLM call in {backoff_time:.2f} seconds...")
                await asyncio.sleep(backoff_time)
                retries += 1
        # Should not be reached if logic is correct, but as a fallback:
        raise Exception(f"LLM call failed after max retries. Last error: {last_exception}")


    @abstractmethod
    async def _execute_llm_call(self, prompt_data: dict) -> str:
        # This method will be implemented by concrete classes like GeminiClientWrapper, OpenAIClientWrapper
        # It will use the specific SDK for the LLM provider.
        pass

    @abstractmethod
    def _is_retryable_error(self, error_exception) -> bool:
        # Logic to check if the caught exception represents a retryable error
        # e.g., check for HTTP status codes 429, 500, 503 from the specific LLM SDK's exceptions
        # This needs to be implemented based on the actual exceptions raised by the LLM SDKs
        # For example, for OpenAI: from openai import RateLimitError, APIError
        # if isinstance(error_exception, (RateLimitError, APIError with status 5xx)): return True
        pass
Tool Implementation: Implement WebSearchTool as described in Section 8.Agent Implementation:For each agent defined in Section 5:Create the Python class, inheriting from ADK's base Agent or LLMAgent.Implement the __init__ method to accept necessary parameters (e.g., configured LLM client instances/wrappers, tools).Implement the run(self, session_state: SessionState, **kwargs) -> SessionState method (or equivalent based on ADK version).Inside run, perform logic: read context from session_state, call LLMs (via the rate-limited client wrapper), invoke tools, update session_state with results and status.Adhere strictly to the SessionState contract (Section 7).Ensure prompts are loaded and formatted correctly.Orchestration Logic:Write the main script (e.g., src/main.py) that defines the full pipeline using InitializerAgent, ParallelAgent, BranchAgent, and SequentialAgent as outlined in Section 4.Correctly configure the BranchAgent's condition based on SessionState values.Implement the loop for max_rounds within the MainDebatePipelineAgent (which itself would be a SequentialAgent orchestrating multiple SingleRoundAgent executions).Testing:Unit Tests: Test each agent in isolation. Mock SessionState, LLM calls (mock the LLM client wrapper), and tool outputs. Verify SessionState transformations. Test rate limiting/retry logic by mocking LLM client to throw errors.Integration Tests: Use ADK's InMemoryRunner (or equivalent) to test sequences of agents and the full pipeline. Verify SessionState flow and conditional logic. Monitor for actual rate limit issues during longer runs.Test edge cases: 0 rounds, stances being identical, tool errors, LLM API errors, non-retryable errors.Rate Limiting & Error Handling (System-Wide):Ensure the LLM client wrappers (Step 3) are used by all LLMAgents.Concurrency Control for ParallelAgent: If ParallelAgent executes multiple LLMAgents, this could lead to a burst of requests. Consider:If ADK offers a way to limit concurrency within ParallelAgent's tasks.Implementing an asyncio.Semaphore shared by LLM-calling agents if they are part of a parallel execution, to limit simultaneous outstanding LLM requests globally or per LLM provider.Staggering calls even within parallel agents if necessary, though the individual agent delays should help.Provider-Specific Error Codes: Ensure _is_retryable_error in the LLM client wrapper correctly identifies retryable vs. non-retryable errors based on specific SDK exceptions for Gemini, OpenAI, Anthropic, etc.Graceful Failure: If an agent fails critically after retries, it should update SessionState with an error status (e.g., session_state["error_message"] = "...", session_state["debate_outcome"] = "Error: Agent X failed"), allowing the pipeline to potentially terminate gracefully or take an error path.10. ADK Best Practices & Anti-Patterns (FAQ for "Windsurf")"Windsurf, pay close attention to this section to ensure your generated code aligns with ADK best practices."Best Practices:Explicit SessionState Usage: Be deliberate about what goes into and comes out of SessionState. This is your primary communication channel. Document this contract (Section 7).Agent Independence (Conditioned by SessionState): Agents should be runnable independently given the correct SessionState input. Their logic shouldn't depend on side effects from non-orchestrated prior agents.Use ADK Orchestrators: Leverage SequentialAgent, ParallelAgent, and BranchAgent for flow control rather than custom Python loops managing agent calls directly in most cases.Stateless Agents (where possible): Agents should primarily rely on SessionState for inputs and outputs, rather than maintaining complex internal instance state across run calls within a single pipeline execution.Idempotency: If an agent's run method can be called multiple times with the same SessionState and produce the same result without unintended side effects, it's more robust. (Not always fully achievable with LLMs, but a good principle).Thorough Logging: Add logging within agents (and LLM client wrappers) to trace execution, SessionState changes, LLM calls, and tool usage. This is invaluable for debugging.Configuration Over Hardcoding: Externalize configurations like model names, prompt templates (if very long), delay times, retry counts, etc., possibly using a configuration file loaded into SessionState or passed to agents.Modular Design: Keep agents focused on a single, well-defined task.Separation of Concerns: Logic for LLM interaction, tool use, and state updates should be clearly organized within an agent.Anti-Patterns & Solutions (Common Errors to Avoid):AP1: The Monolith Agent (Overly complex single agent)Symptom: One agent doing initialization, debating, moderating, and evaluating.Problem: Hard to test, debug, and reuse. Violates modularity.Solution: Decompose into smaller, specialized agents as outlined in this plan.AP2: Chatty Agents with Direct Coupling (Agents directly calling methods of other agent instances)Symptom: Agent A has an instance of Agent B and calls agent_b.do_something().Problem: Creates tight coupling, makes orchestration complex, bypasses ADK's intended state flow.Solution: Use SessionState for all inter-agent communication. Agent A writes to SessionState, Agent B (when run by an orchestrator) reads from SessionState.AP3: Static Prompts (Hardcoding dynamic values directly in prompt strings)Symptom: prompt = "Debate about bananas. This is round 2."Problem: Not adaptable.Solution: Parameterize prompts (e.g., prompt_template = "Debate about {{topic}}. This is round {{current_round}}.") and use SessionState values to fill them at runtime. ADK's LLMAgent should have built-in support for this.AP4: Reinventing Orchestration Wheels (Manually looping through agent calls instead of using ADK orchestrators)Symptom: A Python for loop in your main script calling agent.run() repeatedly.Problem: Misses out on ADK's optimizations, error handling, and standardized execution patterns.Solution: Use SequentialAgent for sequences, ParallelAgent for concurrency, and BranchAgent for conditions.AP5: Ignoring LLM/Tool Failures (No error handling for external calls)Symptom: An LLM API error or tool failure crashes the entire pipeline without clear status.Problem: Lack of robustness.Solution: Implement try...except blocks around LLM calls (within the client wrapper) and tool invocations. Log errors, update SessionState with an error status, and implement retry strategies as appropriate.AP6: Assuming Tool Magic (LLM hallucinating tool use or tool parameters)Symptom: Prompt says "Use search tool" but the tool isn't configured correctly, or the LLM tries to call it with made-up parameters.Problem: Tool calls will fail or behave unexpectedly.Solution: Ensure tools are correctly defined, instantiated, and passed to the agent (as per ADK docs). The agent's prompt must accurately describe the available tools (name, description, parameters) so the LLM can generate valid requests for tool use.AP7: "Windsurf" Inventing ADK Methods/Parameters (ESPECIALLY for State/Context/Events/LLM Clients/Rate Limiting)Symptom: Code uses agent.non_existent_method() or LLMAgent(made_up_param=True).Problem: Code will not run or will behave unpredictably.Solution: Strictly adhere to the official Google ADK documentation for the specific version being used. Verify all class names, method signatures, constructor parameters, and configuration options against the documentation. Do not assume methods or parameters exist.AP8: Mutable Default Arguments in Agent Constructors or MethodsSymptom: def __init__(self, history=[]) or def run(self, my_list=[]).Problem: The list [] is created once and shared across all instances or calls, leading to unexpected behavior.Solution: Use None as the default and initialize in the method body: def __init__(self, history=None): self.history = history if history is not None else [].AP9: Blocking Operations in Async Agents (If ADK uses an async framework)Symptom: Using time.sleep() in an async def run() method, or making blocking I/O calls without await.Problem: Blocks the event loop, negating benefits of async.Solution: Use asyncio.sleep() for delays in async code. Use async libraries for I/O operations and await them. Ensure LLM client libraries are used in their async versions if available and appropriate.AP10: Ignoring Rate Limits Until They BiteSymptom: Pipeline works for a few calls, then starts failing with HTTP 429 errors.Problem: Insufficient proactive rate limit handling.Solution: Implement delays, retries with exponential backoff, and potentially concurrency limits from the start as detailed in Section 9. Consult the API documentation for each LLM provider (OpenAI, Gemini, Anthropic) for their specific rate limits (RPM, TPM).AP11: Hardcoding LLM Client Initialization/ConfigurationSymptom: self.llm = OpenAI(api_key="...") or fixed model names directly inside an agent.Problem: Not flexible, hard to test, mixes configuration with logic.Solution: LLM clients (or their configurations including model names, API keys, retry settings) should be instantiated outside the agent (e.g., in the main script or a factory based on external config files) and passed into the agent's constructor.AP12: Implicit State Dependencies / "Spooky Action at a Distance"Symptom: An agent's behavior changes based on SessionState modifications made by a distant, non-adjacent agent in the orchestrated flow, making the logic hard to trace. Or an agent modifies a part of SessionState it doesn't "own."Problem: Reduces clarity, increases coupling, makes debugging difficult.Solution: Ensure SessionState modifications are well-defined and primarily affect the context for immediately subsequent agents or conditions in the orchestration. Each piece of state should have a clear "producer." Document your SessionState contract (Section 7 and Section 9, Step 2).AP13: Inventing Custom Event Systems / Bypassing OrchestratorsSymptom: Implementing a complex pub/sub event bus or custom callback registry for agents to signal each other, instead of using ADK orchestrators and SessionState changes.Problem: Fights the ADK framework, adds unnecessary complexity, and misses out on ADK's built-in control flow and state management benefits.Solution: Represent "events" as status flags or data written to SessionState. Use BranchAgent to react to these "events" and direct flow. Use SequentialAgent to ensure an agent runs after another has produced necessary data/status in SessionState.AP14: Misusing Callbacks for Inter-Agent CommunicationSymptom: Agent A tries to pass a callback function to Agent B to be invoked when Agent B completes.Problem: This is not the standard ADK pattern. Orchestrators and SessionState manage this.Solution: Agent B writes its completion status/result to SessionState. The orchestrator then runs Agent A (if sequential) or another part of the flow. Internal callbacks (e.g., for async SDK calls within an agent or tool) are fine, but they should resolve to updating the SessionState that the agent itself returns, or the result the tool returns to the agent.AP15: Agents Mutating Received SessionState Objects Recklessly (If SessionState is mutable and passed by reference)Symptom: If SessionState objects are mutable dictionaries, an agent might inadvertently modify parts of the state it wasn't supposed to, or in a way that breaks expectations for parallel agents (if they share a reference).Problem: Unpredictable behavior, race conditions.Solution: Agents should generally add new keys or update their designated keys in SessionState. If ADK passes deep copies of SessionState to parallel branches, this is less of an issue. Windsurf must verify ADK's behavior for SessionState in ParallelAgent scenarios (see Section 10.1). If state is shared, use caution and clear "ownership" of keys. Consider returning a new or updated copy of the state object if that's the ADK pattern, rather than mutating in place if it causes issues.10.1. Special Consideration: ParallelAgent and SessionStateWindsurf must carefully study ADK documentation on ParallelAgent:State Isolation/Copying: Does each agent in a ParallelAgent receive an independent copy of SessionState, or do they operate on a shared reference? Understanding this is critical to prevent race conditions.State Merging/Fan-In: After parallel execution, how are the SessionState objects (or their modifications) from each branch merged or made available to subsequent agents? ADK might have specific patterns (e.g., a list of results, a dictionary keyed by agent name, or specific merge strategies) or requirements for this.This understanding is crucial to prevent race conditions where parallel agents overwrite each other's state changes or to ensure all relevant results from parallel tasks are correctly consolidated and accessible to downstream agents.Rate limiting for LLM calls within parallel agents also needs careful consideration (see Section 9, Step 8).11. Potential Issues & TroubleshootingSessionState Debugging:Issue: Incorrect data in SessionState or keys missing.Troubleshooting: Log the entire SessionState (or relevant parts) before and after each agent's run() method. Use a debugger to step through. Verify against the SessionState contract (Section 7).LLM Prompt Issues:Issue: LLM output is not in the desired format, irrelevant, or low quality.Troubleshooting:Iteratively refine prompts. Be more specific.Provide examples in the prompt (few-shot prompting).Check that all template variables ({{variable}}) are correctly populated from SessionState.Experiment with LLM temperature and other generation parameters.Ensure the LLM is instructed to use tools when necessary and how to format its request for tool use.Model-Specific Behavior: Prompts that work well with one model (e.g., Gemini) might need adjustment for another (e.g., Anthropic Claude or OpenAI GPT-4). Test prompts with each targeted model.Tool Integration Failures:Issue: Tool not found, tool errors, LLM not using the tool correctly.Troubleshooting:Verify tool registration and passing to the agent as per ADK documentation.Test the tool in isolation (unit test).Ensure the agent's prompt clearly instructs the LLM on the tool's name, purpose, and parameters.Log the exact input to the tool and output from it.Orchestration Flow Not Behaving as Expected:Issue: Agents run in the wrong order, BranchAgent takes the wrong path.Troubleshooting:Carefully review the list of agents passed to SequentialAgent and ParallelAgent.For BranchAgent, thoroughly test the condition logic. Log the SessionState values used in the condition just before the BranchAgent evaluates it.Rate Limit Errors Despite Delays/Retries:Issue: Still hitting API rate limits (e.g., HTTP 429).Troubleshooting:Verify API keys are correct and active for the specific model/provider tier.Check the dashboard of the LLM provider for specific error messages or quota information (RPM, TPM).Increase delay durations (rate_limit_delay in the LLM client wrapper).Adjust retry parameters (more retries, longer backoff).Ensure the ParallelAgent isn't overwhelming the API (see Section 9, Step 8, and Section 10.1).Confirm you are adhering to tokens-per-minute (TPM) and requests-per-minute (RPM) limits for your API tier.Stance Comparison in BranchAgent:Issue: Simple string comparison of stances might be too naive if stances are phrased differently but mean the same thing.Troubleshooting:Start with simple equality/inequality after normalizing text (lowercase, remove punctuation).If needed, consider a more sophisticated comparison:Use an LLM call with a specific prompt to ask if two stances are fundamentally different. This adds cost and latency but might be more robust.Embedding similarity (more complex to implement).Managing Multiple LLM Client Configurations:Issue: Complexity in initializing and passing different LLM clients (OpenAI, Gemini, Anthropic) to various agents.Troubleshooting:Use a configuration file (e.g., YAML, JSON) or environment variables to define which agent role uses which LLM provider/model and its specific settings (API key env var name, model name, rate limit params).Implement a factory pattern that creates and configures the appropriate LLM client wrapper based on this configuration.Ensure API keys are correctly loaded from distinct environment variables for each provider.Context Not Propagating:Issue: An agent fails because required data is missing from SessionState.Troubleshooting: Trace back the SessionState through the orchestrator. Ensure the producer agent correctly wrote the key and that the SessionState contract (Section 7) is being followed. Log SessionState at each step."Events" (State Changes) Not Triggering Conditional Logic:Issue: A BranchAgent doesn't take the expected path.Troubleshooting: Log the SessionState just before the BranchAgent evaluates its condition. Verify the status/event key it's checking is being set as expected by the preceding agent.Tool Context Issues:Issue: A tool behaves incorrectly due to missing or wrong input.Troubleshooting: Inside the agent calling the tool, log the exact context/parameters being passed to the tool's execution method. Ensure these are correctly derived from SessionState.12. Conclusion & Next StepsThis updated project plan provides a more robust and comprehensive roadmap for "Windsurf," particularly concerning the critical aspects of state, context, and event-like signal management using SessionState and ADK orchestrators, along with multi-LLM support and rate limiting. Adherence to these guidelines, coupled with careful consultation of the official ADK documentation, will be key to a successful and idiomatic ADK implementation.Next Steps for "Windsurf":Thoroughly review this entire document.Consult the specific Google ADK version documentation for all mentioned components. Pay special attention to:The exact API and behavior of SessionState (how it's passed, mutability, copying semantics, especially with ParallelAgent).How LLMAgent (or equivalent) is designed to work with different LLM providers and how it ingests context from SessionState for prompts.The mechanisms by which SequentialAgent, ParallelAgent, and BranchAgent manage and propagate SessionState.Any built-in ADK utilities for common state operations, rate limiting, or async patterns.Tool definition, registration, and invocation within agents.Implement Step 2 from Section 9 (Define SessionState Contract EXPLICITLY) before writing agent code. This is a foundational step.Proceed with other implementation steps (environment setup, LLM client wrappers, tools, agents, orchestration) as outlined, always referring back to the SessionState contract and ADK documentation.Prioritize robust error handling and logging from the beginning.Remember, SessionState is the lifeblood of communication and context in this ADK application. Using it correctly, as defined by ADK patterns, is paramount.



Windsurf Gemini 2.5 Outpu:

Based on these documents, here's a comprehensive plan for building the multi-agent debate engine using Google ADK, keeping in mind your Windows PowerShell environment and the requirement for Python within a venv.

This plan is designed to be followed systematically.

Phase 1: Project Setup & Foundation

Environment Setup (Windows & PowerShell)
Python & Pip:
Verify Python (3.8+ recommended) and Pip are installed and added to PATH.
In PowerShell: python --version, pip --version
Project Directory:
Create the main project folder (e.g., adk-debate-engine).
In PowerShell: mkdir adk-debate-engine, cd adk-debate-engine
Virtual Environment (venv):
Create the virtual environment: python -m venv .venv
Activate the virtual environment: .\.venv\Scripts\Activate.ps1
Note: If script execution is disabled, you might need to adjust PowerShell's execution policy (e.g., Set-ExecutionPolicy RemoteSigned -Scope CurrentUser or Set-ExecutionPolicy Unrestricted -Scope Process). Understand the security implications.
.gitignore File:
Create a .gitignore file.
Add entries: .venv/, __pycache__/, *.pyc, .env, any IDE-specific folders.
.env File for API Keys:
Create an empty .env file in the project root.
Add placeholder API keys (you'll fill these later):
CopyInsert
GEMINI_API_KEY="your_gemini_api_key_here"
OPENAI_API_KEY="your_openai_api_key_here"
ANTHROPIC_API_KEY="your_anthropic_api_key_here"
# Add other API keys (e.g., for search tools) as needed
Ensure .env is in .gitignore.
Install ADK & Core Dependencies
Google ADK:
With the venv activated, install the Google Agent Development Kit.
Crucial: You (Cascade/Windsurf) must find the exact pip package name from the official Google ADK documentation. The getting_started.md uses google-adk as a placeholder.
Example: pip install <actual-google-adk-package-name>
Supporting Libraries:
Install LLM provider SDKs and python-dotenv: pip install python-dotenv openai google-generativeai anthropic
Adjust based on the final choice of SDKs and any other utility libraries needed.
Project Structure
Create the following directory structure within adk-debate-engine as suggested in getting_started.md:
CopyInsert
adk-debate-engine/
├── .venv/
├── src/
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── initializer_agent.py
│   │   ├── stance_agents.py        # For ProStanceAgent, ConStanceAgent
│   │   ├── debater_agents.py       # For ProDebaterAgent, ConDebaterAgent
│   │   ├── moderator_agent.py
│   │   ├── evaluator_agents.py     # For FinalEvaluatorAgent, ScoreAggregationAgent, EndDebateEarlyAgent
│   │   └── base_llm_agent.py     # Optional: if creating a custom base for LLMAgents with shared logic
│   ├── tools/
│   │   ├── __init__.py
│   │   └── web_search_tool.py
│   ├── core/                     # For LLM client wrappers, SessionState definition, etc.
│   │   ├── __init__.py
│   │   ├── llm_clients.py          # LLM client wrappers and rate limiting
│   │   └── session_state_contract.py # Explicit definition or constants for SessionState keys
│   ├── prompts/                  # To store larger prompt templates
│   │   ├── __init__.py
│   │   ├── pro_stance_prompt.txt
│   │   ├── con_stance_prompt.txt
│   │   ├── pro_debater_prompt.txt
│   │   ├── con_debater_prompt.txt
│   │   ├── moderator_prompt.txt
│   │   └── final_evaluator_prompt.txt
│   ├── __init__.py
│   └── main.py                   # Main script to define and run the debate pipeline
├── .env
├── .gitignore
└── README.md
Create initial __init__.py files in src, src/agents, src/tools, src/core, src/prompts.
Define SessionState Contract (Crucial First Step - proposal.md Section 7 & 9.2)
In src/core/session_state_contract.py (or similar), explicitly define all keys that will be used in SessionState. This is the central "contract" for data flow.
For each key, document: its purpose, data type, which agent(s) produce it, and which agent(s) consume it.
Example keys from proposal.md:
topic (str)
max_rounds (int)
current_round (int)
pro_stance (str), pro_stance_status (str)
con_stance (str), con_stance_status (str)
debate_history (list)
pro_arguments_by_round (dict)
con_arguments_by_round (dict)
round_scores (dict)
final_evaluation_pro (dict), final_evaluation_con (dict)
winner_determination (str), final_reasoning (str)
final_weighted_score_pro (float), final_weighted_score_con (float)
debate_outcome (str)
error_message (str, optional)
last_agent_status (dict, optional)
Phase 2: Core Components Development

LLM Client Configuration & Wrappers (src/core/llm_clients.py)
Refer to proposal.md Section 9, Step 3.
Investigate how ADK's LLMAgent ingests LLM configurations for Gemini, OpenAI, Anthropic.
If ADK doesn't provide a unified, rate-limited interface, create wrapper classes (e.g., BaseLLMClient, GeminiClientWrapper, OpenAIClientWrapper, AnthropicClientWrapper).
Implement Robust Rate Limiting:
Client-side delays (e.g., asyncio.sleep(CONFIGURED_DELAY)). Make delay configurable.
Automatic retry mechanism with exponential backoff and jitter for transient errors (HTTP 429, 5xx). Make retries configurable.
Provider-Specific Error Handling: Implement _is_retryable_error methods in wrappers to correctly identify retryable errors based on specific SDK exceptions for each LLM provider.
API keys should be loaded from environment variables (os.getenv) within these wrappers/configurations.
Tool Implementation (src/tools/web_search_tool.py)
Refer to proposal.md Section 8.
Create WebSearchTool class inheriting from ADK's Tool base class (verify exact base class and methods from ADK docs).
Implement the tool's execution method (e.g., execute or _run).
This method will take a query, use a search API (e.g., Google Custom Search), and return formatted results.
Manage API keys for the search tool (via constructor, loaded from .env).
Handle errors and cases with no results.
The tool's name and description attributes should be clear for LLM consumption.
Phase 3: Agent Implementation (src/agents/)

General Agent Structure (from proposal.md Section 5):
Each agent class will inherit from ADK's Agent or LLMAgent (verify from ADK docs).
__init__: Accept necessary parameters (LLM client instances/configs, tools).
run(self, session_state: SessionState, **kwargs) -> SessionState:
Read context from session_state (adhering to the SessionState contract).
Perform agent logic (LLM calls via rate-limited wrappers, tool invocations).
Publish results/status to session_state.
Return updated session_state.
Load and format prompts correctly (see Step 8).
Implement Each Agent:
a. InitializerAgent (initializer_agent.py)
Type: Agent (custom Python logic).
Input params: topic, max_rounds.
SessionState Writes: topic, max_rounds, current_round = 0, debate_history = [], round_scores = {}, pro_stance = None, etc. (as per contract).
b. ProStanceAgent & ConStanceAgent (stance_agents.py)
Type: LLMAgent.
LLM Config: Configurable (e.g., Gemini for Pro, OpenAI for Con). Pass appropriate LLM client wrapper.
Tools: Optional WebSearchTool.
SessionState Reads: topic.
SessionState Writes: pro_stance/con_stance, pro_stance_status/con_stance_status.
Prompts: Specific prompts for affirmative/opposing stances.
c. ProDebaterAgent & ConDebaterAgent (debater_agents.py)
Type: LLMAgent.
LLM Config: Configurable.
Tools: Mandatory WebSearchTool.
SessionState Reads: topic, current_round, pro_stance/con_stance, debate_history, opponent's previous argument from con_arguments_by_round/pro_arguments_by_round.
SessionState Writes: pro_arguments_by_round[current_round]/con_arguments_by_round[current_round], updates debate_history.
Prompts: User-provided "Debate Agent Prompts".
d. ModeratorAgent (moderator_agent.py)
Type: LLMAgent.
LLM Config: Configurable.
Tools: N/A.
SessionState Reads: topic, current_round, pro_arguments_by_round[current_round], con_arguments_by_round[current_round], debate_history.
SessionState Writes: round_scores[current_round] (with scores, justifications, summary), updates debate_history.
Prompts: User-provided "Moderator Prompt".
e. FinalEvaluatorAgent (evaluator_agents.py)
Type: LLMAgent.
LLM Config: Configurable (powerful model recommended).
Tools: N/A.
SessionState Reads: topic, debate_history (complete), round_scores (all).
SessionState Writes: final_evaluation_pro/con, winner_determination, final_reasoning.
Prompts: User-provided "Final Evaluator Prompt".
f. ScoreAggregationAgent (evaluator_agents.py)
Type: Agent (custom Python logic).
SessionState Reads: round_scores, final_evaluation_pro['score'], final_evaluation_con['score'].
Calculates and Writes: final_weighted_score_pro, final_weighted_score_con using the 0.25/0.75 weighting.
g. EndDebateEarlyAgent (evaluator_agents.py)
Type: Agent (custom Python logic).
SessionState Reads: pro_stance, con_stance.
SessionState Writes: debate_outcome message.
Prompt Engineering & Management (src/prompts/)
Store prompt templates in separate files (e.g., pro_debater_prompt.txt).
Load prompts in agents.
Ensure all placeholders ({{topic}}, {{current_round}}, etc.) are dynamically filled using values from SessionState. ADK's LLMAgent likely supports this.
Follow prompt design principles from proposal.md (clarity, role definition, structured output requests).
Be prepared for model-specific prompt tuning during testing.
Phase 4: Orchestration & Execution

Orchestration Logic (src/main.py)
Refer to proposal.md Section 4 (System Architecture).
Use ADK orchestrators: SequentialAgent, ParallelAgent, BranchAgent. Verify their exact names and usage from ADK documentation.
Load environment variables using dotenv.load_dotenv().
Instantiate all agents, passing configured LLM clients and tools.
Define the main pipeline:
InitializerAgent.
ParallelAgent for ProStanceAgent and ConStanceAgent.
Crucial (Proposal Section 10.1): Study ADK docs on ParallelAgent state handling (isolation, merging/fan-in).
BranchAgent:
Condition: Checks if pro_stance and con_stance from SessionState are meaningfully different.
if_true: Points to the Main Debate Pipeline.
if_false: Points to EndDebateEarlyAgent.
Main Debate Pipeline (SequentialAgent):
This will likely involve a loop construct or a custom ADK agent that itself orchestrates rounds if ADK doesn't directly support looping N times with state updates. The proposal.md suggests "This agent iterates max_rounds times". Investigate how to implement this loop, potentially by having the SequentialAgent (Main Debate Pipeline) re-trigger itself or by using a controlling agent that invokes the SingleRoundAgent sequentially max_rounds times, updating current_round in SessionState each time.
Each iteration runs a SingleRoundAgent (SequentialAgent):
ProDebaterAgent
ConDebaterAgent
ModeratorAgent
FinalEvaluatorAgent.
ScoreAggregationAgent.
The main script should accept initial parameters like topic and max_rounds.
Execution
The main execution block in src/main.py:
python
CopyInsert
if __name__ == "__main__":
    load_dotenv()
    # Get API Keys (and check if they exist)
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    # ... other keys ...
    if not gemini_api_key: # etc. for all required keys
        print("Error: API key not found. Ensure .env file is set up.")
        exit(1)

    # Initialize ADK runtime/runner (consult ADK docs)
    # ...

    # Instantiate agents with configurations (LLM clients, tools)
    # ...

    # Define the orchestrated pipeline
    # ... pipeline = SequentialAgent([...]) ...

    # Run the pipeline with initial SessionState
    # initial_state = SessionState(topic="AI Ethics", max_rounds=3) # Or however ADK SessionState is created
    # final_state = adk_runner.run(pipeline, initial_state) # Consult ADK docs

    # Print final_state or relevant results
    # print(final_state)
Run from PowerShell (with venv activated, from the project root): python src/main.py
Phase 5: Testing & Refinement

Testing Strategy
Unit Tests:
Test each agent in isolation (e.g., using Python's unittest or pytest).
Mock SessionState inputs.
Mock LLM client wrapper calls (to avoid actual API calls and test specific responses/errors).
Mock tool outputs.
Verify SessionState transformations and outputs.
Test rate-limiting/retry logic in LLM wrappers by mocking client errors.
Integration Tests:
Use ADK's in-memory runner (if available, check docs) to test sequences of agents and the full pipeline.
Verify SessionState flow between agents and conditional logic of BranchAgent.
Monitor for actual rate limit issues during short test runs (then rely on robust client-side handling).
Edge Case Testing: 0 rounds, identical stances, tool errors, LLM API errors, non-retryable errors.
ADK Best Practices & Anti-Patterns Review
Continuously refer to proposal.md Section 10 ("ADK Best Practices & Anti-Patterns").
Emphasize: Explicit SessionState usage, agent independence (via SessionState), use of ADK orchestrators, stateless agents where possible, thorough logging, configuration over hardcoding.
Avoid: Monolith agents, direct agent coupling, static prompts, reinventing orchestration, ignoring failures, etc.
Crucial: Always prefer ADK's built-in mechanisms for state, context, events, and control flow over inventing custom solutions.
General Considerations:

ADK Documentation is Key: This plan relies heavily on the structures proposed in proposal.md. However, the actual implementation details (class names, method signatures, specific ADK primitives for orchestration, state management, tool usage, LLM integration) must come from the official Google ADK documentation. You (Cascade/Windsurf) need to treat the ADK docs as the ultimate source of truth for API specifics.
Logging: Implement comprehensive logging throughout the agents and core components. This will be invaluable for debugging.
Iterative Development: Build and test components incrementally. Start with the InitializerAgent and a simple SessionState, then add the stance agents, then one round of debate, etc.
This plan should provide a solid roadmap. Remember to consult the Google ADK documentation at each step for the precise implementation details.




IMPORTANT NOTES
REVIEW @WEB GOOGLE ADK DOCUMENTATION WHEN NECESSARY TO VALIDATE METHODS, PARAMETERS, ETC.
USE PYTHON -C TO VALIDATE FROM LOCAL AS NEEDED