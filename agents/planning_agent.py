import os
from utils.llm_integration import call_gemini

class PlanningAgent:
    def __init__(self):
        pass

    def generate_plan(self, symbol: str, memory: str = None) -> list[str]:
        """Generates a research plan for a given stock symbol."""
        
        system_instruction = (
            "You are an expert investment analyst planning a research workflow. "
            "Given the stock symbol and the historical memory, generate a list of the 5-7 most critical steps "
            "(including tool calls and internal processes) to generate a final investment thesis. "
            "Output must be a JSON array of strings."
        )
        
        user_prompt = f"Stock Symbol: {symbol}"
        if memory:
            user_prompt += f"\n\nHistorical Memory:\n{memory}"
            
        response = call_gemini(system_instruction, user_prompt, json_output=True)
        
        if response and isinstance(response, list):
            return response
        else:
            print("Failed to generate a valid plan.")
            return []
