import os
import dotenv
import google.generativeai as genai
from openai import OpenAI
from anthropic import Anthropic

# Load environment variables from .env file
dotenv.load_dotenv()

# --- Environment Variable Keys ---
GEMINI_API_KEY_ENV = "GEMINI_API_KEY"
OPENAI_API_KEY_ENV = "OPENAI_API_KEY"
ANTHROPIC_API_KEY_ENV = "ANTHROPIC_API_KEY"
GOOGLE_CLOUD_PROJECT_ENV = "GOOGLE_CLOUD_PROJECT"
GOOGLE_CLOUD_LOCATION_ENV = "GOOGLE_CLOUD_LOCATION"
GOOGLE_GENAI_USE_VERTEX_AI_ENV = "GOOGLE_GENAI_USE_VERTEXAI"  # Expected 'True' or 'False'

# --- Client Initialization ---

_gemini_client_configured = False # Tracks if genai.configure has been called
_openai_client = None
_anthropic_client = None

def get_gemini_client():
    """
    Ensures Google Gemini is configured and returns the genai module.
    Supports Vertex AI if GOOGLE_GENAI_USE_VERTEXAI is 'True' and project/location are set.
    Raises ValueError if API key or necessary Vertex AI config is missing.
    """
    global _gemini_client_configured
    if not _gemini_client_configured:
        api_key = os.getenv(GEMINI_API_KEY_ENV)
        use_vertex_ai_str = os.getenv(GOOGLE_GENAI_USE_VERTEX_AI_ENV, "False").lower()
        use_vertex_ai = use_vertex_ai_str == "true"

        if use_vertex_ai:
            project_id = os.getenv(GOOGLE_CLOUD_PROJECT_ENV)
            location = os.getenv(GOOGLE_CLOUD_LOCATION_ENV)
            if not project_id or not location:
                raise ValueError(
                    f"To use Vertex AI for Gemini, '{GOOGLE_CLOUD_PROJECT_ENV}' and "
                    f"'{GOOGLE_CLOUD_LOCATION_ENV}' must be set in the .env file."
                )
            # For Vertex AI, authentication is typically handled via Application Default Credentials (ADC).
            # The API key might be optional or ignored if ADC is correctly set up.
            genai.configure(
                api_key=api_key,  # Pass along, might be needed for some Vertex setups or ignored if ADC is primary
                cloud_project=project_id,
                cloud_location=location,
            )
            print(f"Configured Gemini client to use Vertex AI (Project: {project_id}, Location: {location})")
        else:
            if not api_key:
                raise ValueError(f"'{GEMINI_API_KEY_ENV}' not found in .env file for Google AI Studio (non-Vertex AI usage)." )
            genai.configure(api_key=api_key)
            print("Configured Gemini client to use Google AI Studio.")
        
        _gemini_client_configured = True
    return genai # Return the configured genai module to be used for model instantiation

def get_openai_client():
    """
    Initializes and returns an OpenAI client.
    Raises ValueError if API key is missing.
    """
    global _openai_client
    if _openai_client is None:
        api_key = os.getenv(OPENAI_API_KEY_ENV)
        if not api_key:
            raise ValueError(f"'{OPENAI_API_KEY_ENV}' not found in .env file.")
        _openai_client = OpenAI(api_key=api_key)
        print("Initialized OpenAI client.")
    return _openai_client

def get_anthropic_client():
    """
    Initializes and returns an Anthropic client.
    Raises ValueError if API key is missing.
    """
    global _anthropic_client
    if _anthropic_client is None:
        api_key = os.getenv(ANTHROPIC_API_KEY_ENV)
        if not api_key:
            raise ValueError(f"'{ANTHROPIC_API_KEY_ENV}' not found in .env file.")
        _anthropic_client = Anthropic(api_key=api_key)
        print("Initialized Anthropic client.")
    return _anthropic_client

if __name__ == '__main__':
    print("Attempting to initialize LLM clients...")
    
    # Test Gemini
    try:
        gemini_module = get_gemini_client()
        print(f"Gemini configured successfully. Use `gemini_module.GenerativeModel(...)`.")
        # Example: model = gemini_module.GenerativeModel('gemini-1.5-flash-latest')
        # print("Able to reference gemini.GenerativeModel.")
    except ValueError as e:
        print(f"Error configuring Gemini: {e}")
    except Exception as e:
        print(f"An unexpected error occurred with Gemini configuration: {e}")

    # Test OpenAI
    try:
        openai_client = get_openai_client()
        if openai_client:
            print(f"OpenAI client initialized: {type(openai_client)}")
            # Test with a simple call like listing models (optional, incurs token usage)
            # models = openai_client.models.list()
            # print(f"Successfully listed OpenAI models (first few): {[m.id for m in models.data[:3]]}")
    except ValueError as e:
        print(f"Error initializing OpenAI client: {e}")
    except Exception as e:
        print(f"An unexpected error occurred with OpenAI client: {e}")

    # Test Anthropic
    try:
        anthropic_client = get_anthropic_client()
        if anthropic_client:
            print(f"Anthropic client initialized: {type(anthropic_client)}")
            # Test with a simple call like counting tokens (optional, incurs token usage)
            # token_count = anthropic_client.count_tokens("Hello world!")
            # print(f"Successfully counted tokens with Anthropic client: {token_count}")
    except ValueError as e:
        print(f"Error initializing Anthropic client: {e}")
    except Exception as e:
        print(f"An unexpected error occurred with Anthropic client: {e}")

    print("LLM client initialization tests completed.")
