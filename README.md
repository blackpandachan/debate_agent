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
| `fact_checker` | `Agent` + `google_search` (always Gemini) | Verifies the round's 3–5 most consequential claims: Supported / Partly true / False / Unverifiable, with its own sources. |
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
| `JUDGE_MODEL` | `DEBATE_MODEL` | Moderator and final judge. Set `claude-haiku-5-5` for a Claude judge. |
| `FACT_CHECK_MODEL` | `JUDGE_MODEL` if Gemini, else `DEBATE_MODEL` | Fact-checker (must be Gemini: it uses Google Search). |
| `MAX_DEBATE_ROUNDS` | `3` | Used when session state has no `max_rounds` (e.g. in `adk web`). |
| `ENABLE_SEARCH` | `true` | Google Search grounding for debaters (Gemini only). |
| `ENABLE_FACT_CHECK` | `true` | Per-round fact-checker (needs search). |
| `LLM_RETRY_ATTEMPTS` | `5` | HTTP retries on transient Gemini errors. |

### Claude as the judge

```bash
JUDGE_MODEL=claude-haiku-5-5 python main.py "Is a hot dog a sandwich?" --rounds 2
```

This also needs `ANTHROPIC_API_KEY`. `claude-*` models use ADK's native
`AnthropicLlm`, which calls the Anthropic API directly, so LiteLLM isn't needed.
Gemini still runs the stances, debaters and fact-checker, since those use Google
Search. Claude Haiku runs the moderator and final judge. Using a different
provider to score than to argue reduces the chance of a judge favoring text
written in its own style.

Claude agents run with adaptive thinking, so Haiku reasons before it scores. ADK
offers the output-schema tool instead of forcing it, because forced tool use isn't
allowed while thinking is on. Either way the reply is checked against the schema,
and a malformed reply is handled by the node retry.

If you set `DEBATE_MODEL` to a Claude model too, the debaters run without search
and the fact-checker is skipped. Other providers' model ids use a `/` prefix (for
example `openai/gpt-5`) and go through ADK's `LiteLlm` wrapper; install `litellm`
and set that provider's API key.

## Cost

Every finished debate ends with an **Estimated cost** table: calls, tokens and
USD per model, plus Google Search queries. The same figures are in
`debate_result.json` (`estimated_cost_usd`), in session state
(`cost_estimate`), and in the last line of the CLI output. The estimate uses
list prices from `debate_agent/pricing.py`:

| | Input / 1M tokens | Output / 1M tokens (incl. thinking) |
|---|---|---|
| `gemini-3.5-flash` | $1.50 | $9.00 |
| `claude-haiku-5-5` | $0.10 | $0.50 |
| Google Search grounding | $14 per 1,000 queries (first 5,000 a month free) | |

The estimate ignores the free search allowance, so a light month of play costs
less than it shows. A 2-round all-flash debate makes about 11 model calls, uses
roughly 60k tokens and runs about 15 searches, which comes to roughly $0.40 at
list price. Update `pricing.py` when prices change.

## Tests

The tests replace the models with fakes, so they need no API keys. They cover the
full graph, early exit on consensus, the no-search path, crash-and-resume,
citation placement, and a Claude Haiku judge. The Claude test runs ADK's real
`AnthropicLlm` and the Anthropic SDK against a mocked HTTP transport and checks
the exact request sent:

```bash
pytest
```

## Project layout

```
debate_agent/        ADK agent package (adk web / adk run discover it)
  agent.py           Workflow graph, function nodes, App (root_agent, app)
  evidence.py        Grounding metadata -> citations + evidence log
  pricing.py         Price table and per-debate cost estimate
  schemas.py         Pydantic models: node I/O and the session-state schema
  prompts.py         Agent instructions
  config.py          Model / env configuration
main.py              CLI: run, list, resume, export
tests/               Offline workflow tests
```
