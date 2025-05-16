Getting Started: ADK Multi-Agent Debate EngineThis guide will walk you through setting up your development environment on Windows using PowerShell and Python's venv for the ADK Multi-Agent Debate Engine project.Target Audience: Developer (and "Windsurf" AI Agent)Environment: Windows, PowerShell, Python 3.x1. PrerequisitesPython: Ensure you have Python installed (Python 3.8 or newer is recommended). You can download it from python.org. During installation, make sure to check the box that says "Add Python to PATH."Pip: Pip is the Python package installer and is usually installed automatically with Python.Windows PowerShell: This is typically available by default on Windows.You can verify your Python and pip installation by opening PowerShell and typing:python --version
pip --version
2. Project Directory SetupCreate a Project Folder:Open PowerShell and navigate to where you want to create your project (e.g., C:\Users\YourUser\Documents\Projects). Then create a directory for the debate engine:cd C:\Users\YourUser\Documents\Projects
mkdir adk-debate-engine
cd adk-debate-engine
Initial Folder Structure (Suggestion):Inside adk-debate-engine, you can create a few subdirectories to organize your code. This is a suggestion, and you can adapt it as the project grows.adk-debate-engine/
├── .venv/                   # Virtual environment (will be created)
├── src/                     # Source code for your agents, tools, etc.
│   ├── agents/              # Agent definitions
│   │   ├── __init__.py
│   │   ├── initializer_agent.py
│   │   ├── stance_agents.py
│   │   ├── debater_agents.py
│   │   ├── moderator_agent.py
│   │   └── evaluator_agent.py
│   ├── tools/               # Tool definitions
│   │   ├── __init__.py
│   │   └── search_tool.py
│   ├── core/                # Core logic, LLM client wrappers, etc.
│   │   └── __init__.py
│   ├── __init__.py
│   └── main.py              # Main script to run the debate pipeline
├── .env                     # For API keys and configurations (DO NOT COMMIT)
├── .gitignore               # To specify intentionally untracked files that Git should ignore
└── README.md                # Project overview
You can create these initial source folders and files as placeholders:mkdir src
mkdir src\agents
mkdir src\tools
mkdir src\core
New-Item src\__init__.py -ItemType File
New-Item src\main.py -ItemType File
New-Item src\agents\__init__.py -ItemType File
New-Item src\tools\__init__.py -ItemType File
New-Item src\core\__init__.py -ItemType File
New-Item .env -ItemType File
New-Item .gitignore -ItemType File
New-Item README.md -ItemType File
3. Create and Activate Virtual Environment (venv)Using a virtual environment is crucial to manage project-specific dependencies.Create the Virtual Environment:While in your project's root directory (adk-debate-engine), run the following command in PowerShell:python -m venv .venv
This creates a folder named .venv containing the Python interpreter and a copy of pip.Activate the Virtual Environment:To start using the virtual environment, you need to activate it. In PowerShell, the command is:.\.venv\Scripts\Activate.ps1
Troubleshooting: If you get an error about script execution being disabled, you might need to change the execution policy for your current session. You can do this by running PowerShell as Administrator and then executing Set-ExecutionPolicy RemoteSigned -Scope CurrentUser or, for the current session only, Set-ExecutionPolicy Unrestricted -Scope Process. Be sure to understand the security implications before changing execution policies.Once activated, your PowerShell prompt should change to indicate the active environment (e.g., (.venv) PS C:\Users\YourUser\Documents\Projects\adk-debate-engine>).4. Install Google Agent Development Kit (ADK)With your virtual environment activated, you can now install the ADK.Find the correct package name: "Windsurf" will need to consult the official Google ADK documentation for the exact pip package name. It might be something like google-adk, agent-development-kit, or similar.Installation command (example):pip install google-adk # Replace 'google-adk' with the actual package name
Also, install any other necessary libraries, such as specific LLM provider SDKs (e.g., openai, google-generativeai, anthropic) or libraries for handling .env files (e.g., python-dotenv).pip install python-dotenv openai google-generativeai anthropic
5. Environment Variables for API Keys (.env file)Sensitive information like API keys should not be hardcoded. Use a .env file.Edit your .env file:Open the .env file you created in the root of your project and add your API keys:# .env
GEMINI_API_KEY="your_gemini_api_key_here"
OPENAI_API_KEY="your_openai_api_key_here"
ANTHROPIC_API_KEY="your_anthropic_api_key_here"
# Add other API keys (e.g., for search tools) as needed
Add .env to .gitignore:Ensure your API keys are not committed to version control. Open your .gitignore file and add:# .gitignore
.venv/
__pycache__/
*.pyc
.env
Load Environment Variables in Python:You'll use a library like python-dotenv to load these variables in your Python scripts (e.g., in src/main.py or a configuration module).# Example in src/main.py or a config module
import os
from dotenv import load_dotenv

load_dotenv() # Looks for .env file in the current directory or parent directories

gemini_api_key = os.getenv("GEMINI_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")
# ... and so on
6. Basic "Hello ADK" (Conceptual Starting Point)In your src/main.py, you can start with a very simple ADK structure to verify the installation and setup. This is highly conceptual and "Windsurf" must replace this with actual ADK primitives based on the official documentation.# src/main.py
import os
from dotenv import load_dotenv

# --- ADK Specific Imports ---
# These are placeholders! Windsurf must find the correct ADK imports.
# from adk.core import Agent, SessionState, SequentialAgent # Example
# from adk.llm import LLMAgentConfiguration # Example

# --- Load Environment Variables ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# ... other keys

# --- Placeholder Agent (Windsurf: Replace with actual ADK Agent) ---
class MyFirstAgent: # Replace with actual ADK Agent base class
    def __init__(self, name):
        self.name = name
        print(f"Agent {self.name} initialized.")

    # The run method signature will depend on the ADK's Agent base class
    # It typically takes SessionState and returns SessionState
    def run(self, session_state: dict) -> dict: # Replace dict with ADK's SessionState type
        print(f"Agent {self.name} is running.")
        print(f"Received session state: {session_state}")
        session_state[f"{self.name}_message"] = f"Hello from {self.name}!"
        session_state[f"{self.name}_status"] = "completed"
        return session_state

# --- Placeholder Orchestration (Windsurf: Replace with actual ADK Orchestrators) ---
def run_pipeline():
    print("Starting ADK Debate Engine (Placeholder Pipeline)...")

    # Initialize SessionState (replace dict with ADK's SessionState)
    initial_session_state = {"greeting": "Pipeline started"}

    # Create agent instances
    agent1 = MyFirstAgent(name="Initializer")
    agent2 = MyFirstAgent(name="Greeter")

    # Conceptual Sequential Execution
    # Windsurf: Use ADK's SequentialAgent or other orchestrators here
    print("\n--- Running Agent 1: Initializer ---")
    state_after_agent1 = agent1.run(initial_session_state)

    print("\n--- Running Agent 2: Greeter ---")
    state_after_agent2 = agent2.run(state_after_agent1)

    print("\n--- Final Session State ---")
    print(state_after_agent2)
    print("\nPipeline finished.")

if __name__ == "__main__":
    # This basic check ensures API keys are being loaded.
    # Windsurf will need to implement proper LLM client initialization.
    if not GEMINI_API_KEY:
        print("Warning: GEMINI_API_KEY not found in .env file.")

    run_pipeline()
To Run This Placeholder:Navigate to the src directory in your activated virtual environment and run:cd src
python main.py
7. Next Steps for WindsurfConsult ADK Documentation: Immediately find and thoroughly read the official Google ADK documentation. Pay close attention to:The correct package name for installation.The base Agent and LLMAgent classes and their required methods (especially run).The SessionState object: how it's created, passed, and modified.Orchestrators: SequentialAgent, ParallelAgent, BranchAgent.Tool definition and usage.Configuration for LLM clients (Gemini, OpenAI, Anthropic).Implement the SessionState Contract: As per the main project plan (adk_debate_engine_plan_v2), define the SessionState keys and their purpose (Section 9, Step 2 of the plan).Start Implementing Core Components:LLM client wrappers/configuration (if ADK doesn't fully abstract them).The InitializerAgent as the first true ADK agent.Follow the Project Plan: Use the adk_debate_engine_plan_v2 document as the primary guide for building out the agents and orchestration logic.This getting-started guide should provide a solid foundation for Windsurf to begin the development process in a structured and organized manner. Remember to deactivate your virtual environment when you're done working by typing deactivate in PowerShell.