# ADK Multi-Agent Debate Engine

Two LLM debaters argue a topic (say, *"Is a hot dog a sandwich?"*) over several
moderated rounds. They research with Google Search, every claim they ground gets
an inline citation, an independent fact-checker verifies the key claims each
round, a moderator scores the round, and a final judge decides the winner.

Built on the [Google Agent Development Kit](https://github.com/google/adk-python)
(`google-adk` 2.x) and its graph-based `Workflow`.

## How it works

```
START
  -> init_debate                        function node: topic + round config -> state
  -> (pro_stance, con_stance)           LlmAgents, in parallel (fan-out)
  -> collect_stances                    JoinNode (fan-in)
  -> record_stances
  -> pro_debater -> con_debater -> fact_checker -> moderator -> record_round
         ^                                                          |
         +---------------------- "next_round" ----------------------+
                                                                    |
                                       "conclude" -> final_judge -> publish_result
```

| Node | ADK primitive | What it does |
|---|---|---|
| `init_debate` | function node | Takes the user message as the topic; writes `topic`, `max_rounds`, `current_round` with `Event(state=...)`. |
| `pro_stance` / `con_stance` | `Agent` + `output_schema=Stance` | Opening positions, generated concurrently. |
| `collect_stances` | `JoinNode` | Waits for both stances. |
| `pro_debater` / `con_debater` | `Agent` + `google_search` + `after_model_callback` | One argument per round. The callback turns Gemini's grounding metadata into `[n]` citations, a **Sources** list and an `evidence_log` entry. |
| `fact_checker` | `Agent` + `google_search` | Verifies the round's 3–5 most consequential claims: Supported / Partly true / False / Unverifiable, with its own sources. |
| `moderator` | `Agent` + `output_schema=RoundAssessment` | Summarizes, uses the fact-check, scores each side 0–30, detects consensus. |
| `record_round` | function node + routed edges | Appends to the transcript, saves it as an artifact, then routes `next_round` or `conclude` (last round, or consensus). |
| `final_judge` | `Agent` + `output_schema=FinalEvaluation` | 100-point rubric per side, picks PRO / CON / CONSENSUS. |
| `publish_result` | async function node | Weighted score = 25% average round score + 75% final evaluation; saves the artifacts and emits the report. |

Other ADK features used:

- **`state_schema=DebateState`** (`schemas.py`): every session-state key is declared
  and typed, and ADK rejects undeclared or mistyped writes.
- **`App` with `ResumabilityConfig`**: a debate that dies midway (rate limit,
  network, Ctrl-C) resumes from the last completed node. ADK marks this
  feature experimental.
- **`retry_config`** on every LLM node, plus Gemini HTTP retry options for 429/5xx.
- **Artifacts**: `debate_transcript.md` (a new version is saved after every round),
  `debate_result.json` and `evidence.json`.
- **Local storage**: `main.py` uses the same per-agent SQLite session store and
  file artifact store as `adk web`, under `debate_agent/.adk/`. Debates started
  from the CLI show up in the web UI and the reverse.

Search with structured output: Gemini only returns source URLs (grounding
metadata) for plain-text replies. So the agents that search (debaters,
fact-checker) reply in text, and the agents that score (moderator, judge) use
`output_schema` without tools.

## Setup

Requires Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
cp debate_agent/.env.example debate_agent/.env   # add GOOGLE_API_KEY or GEMINI_API_KEY
```

## Running

```bash
python main.py "Is a hot dog a sandwich?" --rounds 2
```

Each finished debate is exported to `debates/<date>-<topic>-<id>.md`, plus
`.result.json` and `.evidence.json`. The CLI also prints the model-call and
token totals.

```bash
python main.py --list                 # past debates, winners, unfinished ones
python main.py --resume <session-id>  # continue an unfinished debate
```

In the ADK dev UI, pick `debate_agent` and send a topic. Past debates, the event
graph, state and artifacts are all browsable:

```bash
adk web
```

You can also use the terminal runner:

```bash
adk run debate_agent
```

Need ideas? Try *"Is cereal a soup?"*, *"Does pineapple belong on pizza?"*,
*"Is a Pop-Tart a ravioli?"*, *"Should the Oxford comma be mandatory?"* or
*"Is water wet?"*.

## Configuration

Set these in `debate_agent/.env` (see `.env.example`):

| Variable | Default | Purpose |
|---|---|---|
| `DEBATE_MODEL` | `gemini-3.5-flash` | Stance and debater agents. |
| `JUDGE_MODEL` | `DEBATE_MODEL` | Fact-checker, moderator and final judge. |
| `MAX_DEBATE_ROUNDS` | `3` | Used when session state has no `max_rounds` (e.g. in `adk web`). |
| `ENABLE_SEARCH` | `true` | Google Search grounding for debaters (Gemini only). |
| `ENABLE_FACT_CHECK` | `true` | Per-round fact-checker (needs search). |
| `LLM_RETRY_ATTEMPTS` | `5` | HTTP retries on transient Gemini errors. |

Model ids containing `/` (for example `openai/gpt-5`) go through ADK's `LiteLlm`
wrapper; install `litellm` and set that provider's API key. Search and
fact-checking are turned off for non-Gemini models.

A 2-round debate on flash takes about two minutes, with around 11 model calls
and roughly 60k tokens.

## Tests

The tests replace the model with a fake, so they need no API key. They cover the
full graph, early exit on consensus, the no-search path, crash-and-resume, and
citation placement:

```bash
pytest
```

## Project layout

```
debate_agent/        ADK agent package (adk web / adk run discover it)
  agent.py           Workflow graph, function nodes, App (root_agent, app)
  evidence.py        Grounding metadata -> citations + evidence log
  schemas.py         Pydantic models: node I/O and the session-state schema
  prompts.py         Agent instructions
  config.py          Model / env configuration
main.py              CLI: run, list, resume, export
tests/               Offline workflow tests
```
