"""Command-line entry point for the debate workflow.

    python main.py --topic "Should generative AI models be open-sourced?" --rounds 3

You can also use the ADK CLI from the repository root:

    adk web            # browser UI, pick "debate_agent"
    adk run debate_agent
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
# Load env before importing the agent: models are configured at import time.
load_dotenv(ROOT / "debate_agent" / ".env")
load_dotenv(ROOT / ".env")

from google.adk.runners import InMemoryRunner  # noqa: E402
from google.genai import types  # noqa: E402

from debate_agent.agent import TRANSCRIPT_ARTIFACT, root_agent  # noqa: E402

APP_NAME = "debate_agent"
USER_ID = "cli_user"
DEFAULT_TOPIC = "Should generative AI models be open-sourced?"
# Agents with output_schema emit raw JSON; the function node after each one
# renders it as Markdown, so skip their own events in the console.
JSON_NODES = {"pro_stance", "con_stance", "moderator", "final_judge"}


def _node_name(event) -> str:
    if event.node_info and event.node_info.path:
        return event.node_info.path.split("/")[-1].split("@")[0]
    return event.author or "?"


async def run_debate(topic: str, rounds: int | None, out: Path | None) -> None:
    runner = InMemoryRunner(agent=root_agent, app_name=APP_NAME)
    initial_state = {"max_rounds": rounds} if rounds else None
    session = await runner.session_service.create_session(
        app_name=APP_NAME, user_id=USER_ID, state=initial_state
    )
    message = types.Content(role="user", parts=[types.Part.from_text(text=topic)])

    async for event in runner.run_async(
        user_id=USER_ID, session_id=session.id, new_message=message
    ):
        if event.partial or not event.content or not event.content.parts:
            continue
        if _node_name(event) in JSON_NODES:
            continue
        text = "".join(p.text or "" for p in event.content.parts if not p.thought)
        if text.strip():
            print(f"\n===== [{_node_name(event)}] =====\n{text}", flush=True)

    artifact = await runner.artifact_service.load_artifact(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=session.id,
        filename=TRANSCRIPT_ARTIFACT,
    )
    if artifact and artifact.text and out:
        out.write_text(artifact.text, encoding="utf-8")
        print(f"\nTranscript saved to {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an ADK multi-agent debate.")
    parser.add_argument("--topic", default=DEFAULT_TOPIC, help="Debate topic.")
    parser.add_argument(
        "--rounds", type=int, default=None, help="Max rounds (default: MAX_DEBATE_ROUNDS or 3)."
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "debate_transcript.md",
        help="Where to write the Markdown transcript.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING)
    asyncio.run(run_debate(args.topic, args.rounds, args.out))


if __name__ == "__main__":
    main()
