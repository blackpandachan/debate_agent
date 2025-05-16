# ADK Multi-Agent Debate Engine

This project implements a multi-agent debate engine using the Google Agent Development Kit (ADK).

## Setup

1.  Ensure Python 3.8+ is installed.
2.  Clone the repository.
3.  Navigate to the project directory: `cd adk_debate_engine`
4.  Create a virtual environment: `python -m venv .venv`
5.  Activate the virtual environment:
    *   PowerShell: `.\.venv\Scripts\Activate.ps1`
    *   CMD: `.venv\Scripts\activate.bat`
    *   Bash/Zsh: `source .venv/bin/activate`
6.  Install dependencies: `pip install -r requirements.txt` (A `requirements.txt` will be added later)
7.  Create a `.env` file in the root directory and add your API keys (see `.env.example` or `proposal.md` for required keys).

## Running the Debate Engine

```bash
python src/main.py --topic "Your Debate Topic" --rounds 3
```
(Command-line arguments to be finalized)
