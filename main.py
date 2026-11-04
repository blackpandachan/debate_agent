"""Command-line entry point for the debate workflow.

    python main.py "Is a hot dog a sandwich?" --rounds 2
    python main.py --list
    python main.py --resume <session-id>

Debates are stored with the same local services `adk web` uses
(``debate_agent/.adk/session.db`` and ``debate_agent/.adk/artifacts``), so every
CLI debate also shows up in the web UI's session list, and its artifacts
(transcript, result, evidence) are browsable there.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import re
import sys
import warnings
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
# Load env before importing the agent: models are configured at import time.
load_dotenv(ROOT / "debate_agent" / ".env")
load_dotenv(ROOT / ".env")
# ResumabilityConfig is flagged experimental by ADK; we rely on it on purpose.
warnings.filterwarnings("ignore", message=r".*\[EXPERIMENTAL\] ResumabilityConfig")

from google.adk import Runner  # noqa: E402
from google.adk.cli.utils.local_storage import (  # noqa: E402
    create_local_artifact_service,
    create_local_session_service,
)
from google.genai import types  # noqa: E402

from debate_agent.agent import (  # noqa: E402
    APP_NAME,
    EVIDENCE_ARTIFACT,
    RESULT_ARTIFACT,
    TRANSCRIPT_ARTIFACT,
    app,
)

USER_ID = "user"  # the user id `adk web` uses, so sessions appear there
DEFAULT_TOPIC = "Is a hot dog a sandwich?"
EXPORT_DIR = ROOT / "debates"
# Agents with output_schema emit raw JSON; the function node after each one
# renders it as Markdown, so skip their own events in the console.
JSON_NODES = {"pro_stance", "con_stance", "moderator", "final_judge"}


def make_runner() -> Runner:
    """A runner backed by the same per-agent local storage as `adk web`."""
    return Runner(
        app=app,
        session_service=create_local_session_service(base_dir=ROOT, per_agent=True),
        artifact_service=create_local_artifact_service(base_dir=ROOT, per_agent=True),
    )


def _node_name(event) -> str:
    if event.node_info and event.node_info.path:
        return event.node_info.path.split("/")[-1].split("@")[0]
    return event.author or "?"


def _new_usage() -> dict[str, int]:
    return {"prompt": 0, "output": 0, "calls": 0}


async def _stream(runner: Runner, usage: dict[str, int], **run_kwargs) -> None:
    """Prints node messages as they arrive and accumulates token usage."""
    async for event in runner.run_async(**run_kwargs):
        if event.usage_metadata and not event.partial:
            usage["calls"] += 1
            usage["prompt"] += event.usage_metadata.prompt_token_count or 0
            usage["output"] += (event.usage_metadata.candidates_token_count or 0) + (
                event.usage_metadata.thoughts_token_count or 0
            )
        if event.partial or not event.content or not event.content.parts:
            continue
        name = _node_name(event)
        if name in JSON_NODES:
            continue
        text = "".join(p.text or "" for p in event.content.parts if not p.thought)
        if text.strip():
            print(f"\n===== [{name}] =====\n{text}", flush=True)


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:50] or "debate"


async def _export(runner: Runner, session_id: str, topic: str) -> None:
    """Copies the session's artifacts to debates/<date>-<topic>-<id>.*"""
    EXPORT_DIR.mkdir(exist_ok=True)
    stem = f"{datetime.now():%Y%m%d}-{_slug(topic)}-{session_id[:8]}"
    written = []
    for name, suffix in (
        (TRANSCRIPT_ARTIFACT, ".md"),
        (RESULT_ARTIFACT, ".result.json"),
        (EVIDENCE_ARTIFACT, ".evidence.json"),
    ):
        part = await runner.artifact_service.load_artifact(
            app_name=APP_NAME, user_id=USER_ID, session_id=session_id, filename=name
        )
        if part is None:
            continue
        data = part.text.encode() if part.text else part.inline_data.data
        path = EXPORT_DIR / f"{stem}{suffix}"
        path.write_bytes(data)
        written.append(path)
    for path in written:
        print(f"Saved {path.relative_to(ROOT)}")


async def _finish(runner: Runner, session_id: str, usage: dict[str, int]) -> None:
    session = await runner.session_service.get_session(
        app_name=APP_NAME, user_id=USER_ID, session_id=session_id
    )
    print(
        f"\nSession {session_id} · {usage['calls']} model calls ·"
        f" {usage['prompt']:,} input / {usage['output']:,} output tokens"
    )
    if session.state.get("winner"):
        await _export(runner, session_id, session.state.get("topic", ""))
    else:
        print(f"Debate did not finish. Resume with: python main.py --resume {session_id}")


async def run_debate(topic: str, rounds: int | None) -> None:
    runner = make_runner()
    session = await runner.session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        state={"max_rounds": rounds} if rounds else None,
    )
    print(f"Session {session.id}")
    message = types.Content(role="user", parts=[types.Part.from_text(text=topic)])
    usage = _new_usage()
    try:
        await _stream(
            runner, usage, user_id=USER_ID, session_id=session.id, new_message=message
        )
    finally:
        await _finish(runner, session.id, usage)


async def resume_debate(session_id: str) -> None:
    """Continues the last invocation of a session that stopped midway.

    ADK replays the session's events, fast-forwards every node that already
    completed, and re-runs only the work that is left.
    """
    runner = make_runner()
    session = await runner.session_service.get_session(
        app_name=APP_NAME, user_id=USER_ID, session_id=session_id
    )
    if session is None:
        sys.exit(f"No session {session_id}. See: python main.py --list")
    if session.state.get("winner"):
        print("That debate already finished; exporting it again.")
        await _export(runner, session_id, session.state.get("topic", ""))
        return
    invocation_id = next(
        (e.invocation_id for e in reversed(session.events) if e.invocation_id), None
    )
    if invocation_id is None:
        sys.exit("That session has no debate to resume.")
    print(f"Resuming {session.state.get('topic', session_id)!r} ({invocation_id})")
    usage = _new_usage()
    try:
        await _stream(
            runner, usage, user_id=USER_ID, session_id=session_id, invocation_id=invocation_id
        )
    finally:
        await _finish(runner, session_id, usage)


async def list_debates() -> None:
    runner = make_runner()
    listing = await runner.session_service.list_sessions(
        app_name=APP_NAME, user_id=USER_ID
    )
    sessions = sorted(listing.sessions, key=lambda s: s.last_update_time, reverse=True)
    if not sessions:
        print("No debates yet.")
        return
    for listed in sessions:
        # list_sessions omits events and may omit state; fetch the full session.
        s = await runner.session_service.get_session(
            app_name=APP_NAME, user_id=USER_ID, session_id=listed.id
        )
        state = s.state if s else {}
        when = datetime.fromtimestamp(listed.last_update_time).strftime("%Y-%m-%d %H:%M")
        done = len(state.get("round_scores", []))
        status = state.get("winner") or f"unfinished ({done}/{state.get('max_rounds', '?')} rounds)"
        print(f"{listed.id}  {when}  {status:<28} {state.get('topic', '(no topic)')}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an ADK multi-agent debate.")
    parser.add_argument("topic", nargs="?", help=f'Debate topic (default: "{DEFAULT_TOPIC}").')
    parser.add_argument("--topic", dest="topic_opt", help=argparse.SUPPRESS)
    parser.add_argument(
        "--rounds", type=int, default=None, help="Max rounds (default: MAX_DEBATE_ROUNDS or 3)."
    )
    parser.add_argument("--resume", metavar="SESSION_ID", help="Continue an unfinished debate.")
    parser.add_argument("--list", action="store_true", help="List saved debates.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.ERROR)
    try:
        if args.list:
            asyncio.run(list_debates())
        elif args.resume:
            asyncio.run(resume_debate(args.resume))
        else:
            topic = args.topic or args.topic_opt or DEFAULT_TOPIC
            asyncio.run(run_debate(topic, args.rounds))
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()
