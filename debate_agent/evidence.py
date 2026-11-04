"""Turns Google Search grounding metadata into citations and an evidence log.

When an agent uses the built-in ``google_search`` tool, Gemini returns
``grounding_metadata``: the queries it ran, the web pages it used
(``grounding_chunks``) and which spans of the answer each page supports
(``grounding_supports``). An ``after_model_callback`` built here:

* inserts ``[n]`` citation markers after each supported span,
* appends a numbered **Sources** list to the reply, so the transcript, the
  moderator and the final judge all see which claims were backed by a real
  retrieved page and which were not,
* appends an ``EvidenceEntry`` to ``state['evidence_log']``.
"""

from __future__ import annotations

from typing import Callable

from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_response import LlmResponse
from google.genai import types

from .schemas import EvidenceEntry, Source


def _sources(metadata: types.GroundingMetadata) -> tuple[list[Source], dict[int, int]]:
    """Web sources in order, plus a map of chunk index -> citation number."""
    sources: list[Source] = []
    numbers: dict[int, int] = {}
    for index, chunk in enumerate(metadata.grounding_chunks or []):
        web = chunk.web
        if not web or not web.uri:
            continue
        sources.append(Source(title=web.title or web.domain or "source", uri=web.uri))
        numbers[index] = len(sources)
    return sources, numbers


def add_citations(
    text: str, metadata: types.GroundingMetadata, numbers: dict[int, int]
) -> str:
    """Inserts ``[n]`` markers at the end of each grounded span.

    Segment indices are UTF-8 byte offsets into the response text.
    """
    encoded = text.encode("utf-8")
    inserts: dict[int, set[int]] = {}
    for support in metadata.grounding_supports or []:
        segment = support.segment
        if not segment or segment.end_index is None:
            continue
        cited = {
            numbers[i] for i in support.grounding_chunk_indices or [] if i in numbers
        }
        if cited and 0 < segment.end_index <= len(encoded):
            inserts.setdefault(segment.end_index, set()).update(cited)
    for end in sorted(inserts, reverse=True):
        marker = "".join(f"[{n}]" for n in sorted(inserts[end]))
        encoded = encoded[:end] + marker.encode("utf-8") + encoded[end:]
    return encoded.decode("utf-8", errors="ignore")


def render_sources(sources: list[Source]) -> str:
    lines = ["", "", "**Sources**"]
    lines += [f"{n}. [{s.title}]({s.uri})" for n, s in enumerate(sources, start=1)]
    return "\n".join(lines)


def make_evidence_recorder(
    speaker: str,
) -> Callable[[CallbackContext, LlmResponse], LlmResponse | None]:
    """Builds an ``after_model_callback`` that records grounding for ``speaker``."""

    def record_evidence(
        callback_context: CallbackContext, llm_response: LlmResponse
    ) -> LlmResponse | None:
        metadata = llm_response.grounding_metadata
        content = llm_response.content
        if llm_response.partial or not metadata or not content or not content.parts:
            return None

        sources, numbers = _sources(metadata)
        entry = EvidenceEntry(
            round=int(callback_context.state.get("current_round", 0) or 0),
            speaker=speaker,
            queries=list(metadata.web_search_queries or []),
            sources=sources,
        )
        log = list(callback_context.state.get("evidence_log") or [])
        callback_context.state["evidence_log"] = [*log, entry.model_dump()]
        if not sources:
            return None

        text_parts = [p for p in content.parts if p.text and not p.thought]
        if not text_parts:
            return None
        # Supports index into the full answer text, so markers are inserted
        # only when that text is a single part; the sources list is always
        # appended to the last text part.
        last = text_parts[-1]
        body = last.text
        if len(text_parts) == 1:
            body = add_citations(body, metadata, numbers)
        replacement = last.model_copy(update={"text": body + render_sources(sources)})
        llm_response.content = types.Content(
            role=content.role,
            parts=[replacement if p is last else p for p in content.parts],
        )
        return llm_response

    return record_evidence
