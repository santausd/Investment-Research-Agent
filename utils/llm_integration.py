import os
import json
import re

# Toggle this to False to use GeminiGenAI (if available)
USE_OLLAMA = False

# Model names / env fallback
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL_NAME', 'llama2')
GEMINI_MODEL = os.getenv('GEMINI_MODEL_NAME', 'gemini-2.5-flash')
NEWS_API_KEY = os.getenv('NEWS_API_KEY', '')

def get_llm():
    """Return an LLM instance. Attempts to return an Ollama LLM if USE_OLLAMA=True,
    otherwise falls back to Gemini GenAI Chat wrapper. Returns None if no client is available."""
    if USE_OLLAMA:
        try:
            from langchain_ollama import OllamaLLM
            return OllamaLLM(model=OLLAMA_MODEL)
        except Exception as e:
            print(f"Failed to load Ollama model ({OLLAMA_MODEL}): {e}")
            print("Falling back to Gemini...\n")
            pass
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=GEMINI_MODEL)
    except Exception as e:
        print(f"Failed to initialize Gemini: {e}")
        return None


def call_llm(system_instruction: str, user_prompt: str, json_output: bool = True, retry: bool = True) -> dict | list | str:
    """
    Calls LLM via LangChain with a system instruction and user prompt.

    Args:
        system_instruction: The system instruction for the model.
        user_prompt: The user's prompt.
        json_output: Whether to expect a JSON output from the model.
        retry: Whether to retry once if JSON parsing fails.

    Returns:
        Parsed JSON (dict/list) if json_output=True and valid JSON is returned,
        otherwise raw text (str).
    """
    try:
        llm = get_llm() 
        
        full_prompt = f"System: {system_instruction}\nUser: {user_prompt}"
        response = llm.invoke(full_prompt)

        # Handle both LangChain AIMessage and plain string responses
        if hasattr(response, "content"):
            text = response.content.strip()
        elif isinstance(response, str):
            text = response.strip()
        else:
            raise TypeError(f"Unexpected response type: {type(response)}")
    
        # Helper function: extract first valid JSON object or array
        def extract_json(text: str):
            patterns = [r"(\{.*?\})", r"(\[.*?\])"]  # non-greedy
            for pattern in patterns:
                matches = re.findall(pattern, text, re.DOTALL)
                for match in matches:
                    try:
                        return json.loads(match)
                    except json.JSONDecodeError:
                        continue
            return None

        if json_output:
            parsed = extract_json(text)
            if parsed is not None:
                return parsed
            elif retry:
                # Retry once with stricter instruction
                retry_prompt = full_prompt + "\nReturn strictly valid JSON ONLY, no explanations."
                retry_response = llm.invoke(retry_prompt)

                # Handle both LangChain AIMessage and plain string responses
                if hasattr(retry_response, "content"):
                    text = response.content.strip()
                elif isinstance(retry_response, str):
                    text = response.strip()
                else:
                    raise TypeError(f"Unexpected response type: {type(retry_response)}")
    
                parsed = extract_json(text)
                if parsed is not None:
                    return parsed
                else:
                    print("Warning: JSON still invalid after retry, returning raw text.")
                    return retry_response
            else:
                print("Warning: No valid JSON found, returning raw text.")
                return text

        return text

    except Exception as e:
        print(f"An error occurred in call_llm: {e}")
        return None
