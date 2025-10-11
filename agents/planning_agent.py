import os
from utils.llm_integration import call_llm

class PlanningAgent:
    def __init__(self):
        pass

    def generate_plan(self, symbol: str, memory: str = None) -> list[str]:
        """Generates a research plan for a given stock symbol."""
        
        system_instruction = (
            "You are an expert investment analyst planning a research workflow. "
            "Given the stock symbol and the historical memory, generate a list of the 5-7 most critical steps "
            "(including tool calls and internal processes) to generate a final investment thesis. "
            "Output must be a JSON array of strings only, lke this:"
            "[\"Step 1: ...\", \"Step 2: ...\"]"
        )
        
        user_prompt = f"Stock Symbol: {symbol}"
        if memory:
            user_prompt += f"\n\nHistorical Memory:\n{memory}"
            
        response = call_llm(system_instruction, user_prompt, json_output=True)
        
        if not response:
            print("Failed to generate a valid plan.")
            return None

        # Handle both JSON and text outputs
        if isinstance(response, list):
            plan = response
        elif isinstance(response, dict) and "steps" in response:
            plan = response["steps"]
        elif isinstance(response, str):
            # Extract plan lines
            lines = response.splitlines()
            plan = [
                line.strip(" -*")
                for line in lines
                if line.strip() and any(c.isdigit() for c in line)
            ]
            if not plan:
                # fallback — take bullet points or any meaningful text lines
                plan = [line.strip(" -*") for line in lines if line.strip()]
        else:
            plan = [str(response)]

        if not plan:
            print("Failed to extract plan from response.")
            return None

        return plan
