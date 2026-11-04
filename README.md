# ADK Multi-Agent Debate Engine

Two LLM debaters argue a topic over several moderated rounds, then a judge scores
the whole debate. Built on the [Google Agent Development Kit](https://github.com/google/adk-python)
(`google-adk` 2.x) using its graph-based `Workflow` primitive.

## How it works

```
START
  -> init_debate                      function node: stores topic / round config in state
  -> (pro_stance, con_stance)         LlmAgents, run in parallel (fan-out)
  -> collect_stances                  JoinNode (fan-in)
  -> record_stances                   function node
  -> pro_debater -> con_debater -> moderator -> record_round
         ^                                          |
         +---------------- "next_round" ------------+
                                                    |
                         "conclude" -> final_judge -> publish_result
```

| Node | ADK primitive | What it does |
|---|---|---|
| `init_debate` | function node | Takes the user message as the topic, writes `topic`, `max_rounds`, `current_round` to state via `Event(state=...)`. |
| `pro_stance` / `con_stance` | `Agent` + `output_schema=Stance` | Opening positions, generated concurrently. |
| `collect_stances` | `JoinNode` | Waits for both stances. |
| `pro_debater` / `con_debater` | `Agent` + `google_search` tool | One argument per round. Prompts read the topic, stances and transcript from state through `{placeholders}`. |
| `moderator` | `Agent` + `output_schema=RoundAssessment` | Summarizes, flags unsupported claims, scores each side 0–30, detects consensus. |
| `record_round` | function node + routed edges | Appends to the transcript, then routes `next_round` (loop) or `conclude` (max rounds reached or consensus). |
| `final_judge` | `Agent` + `output_schema=FinalEvaluation` | Scores each side on a 100-point rubric and picks PRO / CON / CONSENSUS. |
| `publish_result` | async function node | Weighted score = 25% average round score + 75% final evaluation; saves `debate_transcript.md` as a session **artifact** and emits the report. |

Everything the nodes exchange is a Pydantic model (`debate_agent/schemas.py`), so
LLM output is validated by ADK rather than parsed by hand.

## Setup

Requires Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
cp debate_agent/.env.example debate_agent/.env   # then add your GOOGLE_API_KEY
```

## Running

Command line:

```bash
python main.py --topic "Should generative AI models be open-sourced?" --rounds 3
```

The transcript is written to `debate_transcript.md`.

ADK dev UI (from the repository root), then pick `debate_agent` and send the topic
as a message:

```bash
adk web
```

Or the ADK terminal runner:

```bash
adk run debate_agent
```

## Configuration

Set in `debate_agent/.env` (see `.env.example`):

| Variable | Default | Purpose |
|---|---|---|
| `DEBATE_MODEL` | `gemini-3.5-flash` | Stance and debater agents. |
| `JUDGE_MODEL` | `DEBATE_MODEL` | Moderator and final judge. |
| `MAX_DEBATE_ROUNDS` | `3` | Used when session state has no `max_rounds`. |
| `ENABLE_SEARCH` | `true` | Google Search grounding for debaters (Gemini only). |
| `LLM_RETRY_ATTEMPTS` | `5` | Retries on transient Gemini API errors (429 / 5xx). |

Model ids containing `/` (for example `openai/gpt-5`, `anthropic/claude-sonnet-5-5`)
go through ADK's `LiteLlm` wrapper; install `litellm` and set that provider's API key.
Search grounding is turned off automatically for non-Gemini debaters.

## Tests

The tests replace the model with a fake, so they need no API key:

```bash
pytest
```

## Project layout

```
debate_agent/        ADK agent package (adk web / adk run discover it)
  agent.py           Workflow graph, function nodes, root_agent
  schemas.py         Pydantic models passed between nodes
  prompts.py         Agent instructions
  config.py          Model / env configuration
main.py              CLI runner (InMemoryRunner)
tests/               Offline workflow tests
```
