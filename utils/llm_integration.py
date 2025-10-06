import os
import json
import re
#import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI

'''
def call_gemini(system_instruction: str, user_prompt: str, json_output: bool = True) -> dict | str:
    """
    Calls the Gemini API with a system instruction and user prompt.

    Args:
        system_instruction: The system instruction for the model.
        user_prompt: The user's prompt.
        json_output: Whether to expect a JSON output from the model.

    Returns:
        A dictionary if json_output is True, otherwise a string.
    """
    model = genai.GenerativeModel(
        model_name=os.environ.get('GEMINI_MODEL_NAME'),
        generation_config={"response_mime_type": "application/json"} if json_output else None
    )
    
    prompt = f"{system_instruction}\n\n{user_prompt}"

    try:
        response = model.generate_content(prompt)
        if json_output:
            return json.loads(response.text)
        return response.text
    except Exception as e:
        print(f"An error occurred in call_gemini: {e}")
        return None
'''

def call_gemini(system_instruction: str, user_prompt: str, json_output: bool = True, retry: bool = True) -> dict | list | str:
    """
    Calls Gemini via LangChain with a system instruction and user prompt.

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
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY is not set in environment variables.")

        model_name = os.environ.get("GEMINI_MODEL_NAME", "gemini-1.5-flash")

        llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.7)

        full_prompt = f"System: {system_instruction}\nUser: {user_prompt}"
        response = llm.invoke(full_prompt)
        text = response.content.strip()

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
                retry_response = llm.invoke(retry_prompt).content.strip()
                parsed = extract_json(retry_response)
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
        print(f"An error occurred in call_gemini: {e}")
        return None

