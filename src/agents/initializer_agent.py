from google.adk import Agent
from ..core import session_state_contract as contract

class InitializerAgent(Agent):
    """
    Initializes the debate by setting up the topic, number of rounds,
    and other initial parameters in the session state.
    """
    def __init__(self, agent_id: str = "initializer_agent"):
        """
        Initializes the InitializerAgent.

        Args:
            agent_id (str): The unique identifier for this agent instance.
        """
        super().__init__(name=agent_id) # ADK Agent expects 'name'
        # self.name will hold the agent_id

    def execute(self, session_state: dict, **kwargs) -> dict:
        """
        Executes the agent's logic to initialize debate parameters.

        Args:
            session_state (dict): The current state of the session/pipeline.
            **kwargs: Expected to contain 'debate_topic' and 'num_rounds'.
                      Falls back to defaults if not provided.

        Returns:
            dict: Updates to be merged into the session_state.
        """
        print(f"Agent '{self.name}': Executing...")

        debate_topic = kwargs.get("debate_topic", "The future of artificial intelligence.")
        num_rounds = kwargs.get("num_rounds", 3)
        initial_current_round = 1 # Debate rounds are typically 1-indexed

        updates = {
            contract.TOPIC: debate_topic,
            contract.MAX_ROUNDS: num_rounds,
            contract.CURRENT_ROUND: initial_current_round,
            contract.PRO_STANCE: None,
            contract.PRO_STANCE_STATUS: "pending",
            contract.CON_STANCE: None,
            contract.CON_STANCE_STATUS: "pending",
            contract.DEBATE_HISTORY: [],
            contract.PRO_ARGUMENTS_BY_ROUND: {},
            contract.CON_ARGUMENTS_BY_ROUND: {},
            contract.ROUND_SCORES: {},
            contract.FINAL_EVALUATION_PRO: None,
            contract.FINAL_EVALUATION_CON: None,
            contract.WINNER_DETERMINATION: None,
            contract.FINAL_REASONING: None,
            contract.FINAL_WEIGHTED_SCORE_PRO: 0.0,
            contract.FINAL_WEIGHTED_SCORE_CON: 0.0,
            contract.DEBATE_OUTCOME: "pending",
            contract.ERROR_MESSAGE: None,
            contract.LAST_AGENT_STATUS: None,
            contract.DEBATE_TRANSCRIPT_MARKDOWN: f"# Debate Topic: {debate_topic}\n\n",
        }

        print(f"Agent '{self.name}': Initialized debate topic to '{debate_topic}'.")
        print(f"Agent '{self.name}': Set max rounds to {num_rounds}.")
        print(f"Agent '{self.name}': Set current round to {initial_current_round}.")
        print(f"Agent '{self.name}': Initialized all required session state keys.")

        return updates

if __name__ == '__main__':
    print("Testing InitializerAgent...")
    initializer = InitializerAgent()

    # Test with default values
    print("\n--- Test Case 1: Default values ---")
    initial_state_1 = {}
    updated_state_1 = initializer.execute(initial_state_1)
    print(f"Updated state: {updated_state_1}")
    assert updated_state_1[contract.TOPIC] == "The future of artificial intelligence."
    assert updated_state_1[contract.MAX_ROUNDS] == 3
    assert updated_state_1[contract.CURRENT_ROUND] == 1
    assert contract.DEBATE_HISTORY in updated_state_1
    assert updated_state_1[contract.PRO_STANCE] is None
    assert updated_state_1[contract.CON_STANCE] is None
    assert updated_state_1[contract.DEBATE_TRANSCRIPT_MARKDOWN].startswith("# Debate Topic:")

    # Test with provided values
    print("\n--- Test Case 2: Provided values ---")
    initial_state_2 = {}
    custom_topic = "The ethics of gene editing."
    custom_rounds = 5
    updated_state_2 = initializer.execute(initial_state_2, debate_topic=custom_topic, num_rounds=custom_rounds)
    print(f"Updated state: {updated_state_2}")
    assert updated_state_2[contract.TOPIC] == custom_topic
    assert updated_state_2[contract.MAX_ROUNDS] == custom_rounds
    assert updated_state_2[contract.CURRENT_ROUND] == 1

    print("\nInitializerAgent testing finished.")
