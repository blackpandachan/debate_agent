from google.adk.agents import Agent
import logging

# Configure logging
logger = logging.getLogger(__name__)
from core import session_state_contract as contract

class InitializerAgent(Agent):
    """
    Initializes the debate by setting up the topic, number of rounds,
    and other initial parameters in the session state.
    """
    # Declare Pydantic fields for attributes assigned in __init__
    topic: str
    max_rounds: int
    def __init__(self, agent_id: str = "initializer_agent", topic: str = "The future of AI", max_rounds: int = 3):
        """
        Initializes the InitializerAgent.

        Args:
            agent_id (str): The unique identifier for this agent instance.
            topic (str): The topic of the debate.
            max_rounds (int): The maximum number of rounds for the debate.
        """
        # Pass all Pydantic fields to super().__init__
        super().__init__(name=agent_id, topic=topic, max_rounds=max_rounds)
        # Pydantic's __init__ (called via super) handles setting self.topic and self.max_rounds.

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
        print(f"Agent '{self.name}': Executing with simplified debug payload...")
        # Using self.topic and self.max_rounds which are set during __init__
        # Forcing a very simple payload for debugging Pydantic validation
        debug_payload = {
            contract.TOPIC: "Debug Topic via InitializerAgent",
            contract.MAX_ROUNDS: 1, # Simplified for debug
            "debug_initializer_ran_successfully": True
        }
        logger.info(f"InitializerAgent: Executed. Returning simplified debug payload: {debug_payload}")
        return debug_payload

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
