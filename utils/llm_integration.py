import os
import json
import google.generativeai as genai

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

    #print(os.environ.get('GOOGLE_API_KEY'), os.environ.get('GEMINI_MODEL_NAME'))

    genai.configure(
        api_key=os.environ.get('GOOGLE_API_KEY'),
    )
    
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
