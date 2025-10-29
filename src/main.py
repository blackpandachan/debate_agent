from __future__ import annotations

import argparse
import logging
import os
from typing import Any, Dict

from dotenv import load_dotenv

from src.agents.critic_agent import CriticAgent
from src.agents.debater_agent import DebaterAgent
from src.agents.evaluator_agents import FinalEvaluatorAgent, ScoreAggregationAgent
from src.agents.initializer_agent import InitializerAgent
from src.agents.moderator_agent import ModeratorAgent
from src.agents.stance_agents import ConStanceAgent, ProStanceAgent
from src.core import session_state_contract as contract

LOGGER = logging.getLogger(__name__)


def build_agents(llm_service: str, prompts_dir: str) -> Dict[str, Any]:
    pro_stance_prompt = os.path.join(prompts_dir, "pro_stance_prompt.txt")
    con_stance_prompt = os.path.join(prompts_dir, "con_stance_prompt.txt")
    pro_debater_prompt = os.path.join(prompts_dir, "pro_debater_prompt.txt")
    con_debater_prompt = os.path.join(prompts_dir, "con_debater_prompt.txt")
    moderator_prompt = os.path.join(prompts_dir, "moderator_prompt.txt")
    critic_prompt = os.path.join(prompts_dir, "critic_prompt.txt")
    final_evaluator_prompt = os.path.join(prompts_dir, "final_evaluator_prompt.txt")

    initializer = InitializerAgent(agent_id="initializer", topic="", max_rounds=0)
    pro_stance = ProStanceAgent(
        agent_id="pro_stance_agent",
        llm_service_name=llm_service,
        prompt_file_path=pro_stance_prompt,
    )
    con_stance = ConStanceAgent(
        agent_id="con_stance_agent",
        llm_service_name=llm_service,
        prompt_file_path=con_stance_prompt,
    )
    pro_debater = DebaterAgent(
        agent_id="pro_debater",
        side="PRO",
        llm_service_name=llm_service,
        prompt_file_path=pro_debater_prompt,
    )
    con_debater = DebaterAgent(
        agent_id="con_debater",
        side="CON",
        llm_service_name=llm_service,
        prompt_file_path=con_debater_prompt,
    )
    critic = CriticAgent(
        agent_id="critic_agent",
        llm_service_name=llm_service,
        prompt_file_path=critic_prompt,
    )
    moderator = ModeratorAgent(
        agent_id="moderator_agent",
        llm_service_name=llm_service,
        prompt_file_path=moderator_prompt,
    )
    final_evaluator = FinalEvaluatorAgent(
        agent_id="final_evaluator_agent",
        llm_service_name=llm_service,
        prompt_file_path=final_evaluator_prompt,
    )
    aggregator = ScoreAggregationAgent()

    return {
        "initializer": initializer,
        "pro_stance": pro_stance,
        "con_stance": con_stance,
        "pro_debater": pro_debater,
        "con_debater": con_debater,
        "critic": critic,
        "moderator": moderator,
        "final_evaluator": final_evaluator,
        "aggregator": aggregator,
    }


def run_debate(topic: str, rounds: int, llm_service: str, prompts_dir: str) -> Dict[str, Any]:
    agents = build_agents(llm_service, prompts_dir)
    session_state: Dict[str, Any] = {}

    agents["initializer"].execute(session_state, debate_topic=topic, num_rounds=rounds)

    LOGGER.info("Generating initial stances...")
    agents["pro_stance"].execute(session_state)
    agents["con_stance"].execute(session_state)

    max_rounds = session_state.get(contract.MAX_ROUNDS, rounds)

    for round_number in range(1, max_rounds + 1):
        session_state[contract.CURRENT_ROUND] = round_number
        if not session_state.get(contract.DEBATE_CONTINUES, True):
            LOGGER.info("Debate ended early before round %s", round_number)
            break

        LOGGER.info("--- Round %s ---", round_number)
        agents["pro_debater"].execute(session_state)
        agents["con_debater"].execute(session_state)
        agents["critic"].execute(session_state)
        agents["moderator"].execute(session_state)

        if not session_state.get(contract.DEBATE_CONTINUES, True):
            LOGGER.info("Debate terminated after critic review in round %s", round_number)
            break

    LOGGER.info("Final evaluation phase")
    agents["final_evaluator"].execute(session_state)
    agents["aggregator"].run(session_state)

    return session_state


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the debate workflow.")
    parser.add_argument("--topic", type=str, required=True, help="Debate topic/question.")
    parser.add_argument(
        "--rounds", type=int, default=3, help="Maximum number of debate rounds.")
    parser.add_argument(
        "--llm-service",
        type=str,
        default=os.getenv("LLM_SERVICE_MAIN", "gemini"),
        help="LLM service identifier (gemini, openai, anthropic).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)

    project_root = os.path.dirname(os.path.abspath(__file__))
    load_dotenv(dotenv_path=os.path.join(project_root, "..", ".env"))
    prompts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts")

    LOGGER.info(
        "Starting debate: topic=%s | rounds=%s | llm_service=%s",
        args.topic,
        args.rounds,
        args.llm_service,
    )

    session_state = run_debate(args.topic, args.rounds, args.llm_service.lower(), prompts_dir)

    LOGGER.info("Winner: %s", session_state.get(contract.WINNER_DETERMINATION, "Unknown"))
    LOGGER.info("Reasoning: %s", session_state.get(contract.FINAL_REASONING, "No reasoning."))
    LOGGER.info(
        "Scores - PRO: %.2f | CON: %.2f",
        session_state.get(contract.FINAL_WEIGHTED_SCORE_PRO, 0.0),
        session_state.get(contract.FINAL_WEIGHTED_SCORE_CON, 0.0),
    )

    transcript_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "debate_transcript.md")
    with open(transcript_path, "w", encoding="utf-8") as handle:
        handle.write(session_state.get(contract.DEBATE_TRANSCRIPT_MARKDOWN, ""))
    LOGGER.info("Transcript saved to %s", transcript_path)


if __name__ == "__main__":
    main()

