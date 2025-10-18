import os
from utils.llm_integration import call_gemini
from utils.prompt_manager import PromptManager
from utils.logger_config import logger

class PlanningAgent:
    def __init__(self):
        self.prompt_manager = PromptManager()

    def generate_plan(self, symbol: str, memory: str = None) -> list[str]:
        """Generates a research plan for a given stock symbol."""
        
        system_instruction = (
            "You are an expert investment analyst planning a research workflow. "
            "Given a stock symbol and historical memory, generate a structured research plan. "
            "The plan should be a JSON array of objects, where each object represents a step. "
            "Each step must have a 'tool' and 'parameters'. "
            "The 'tool' must be one of the following: ['yfinance', 'newsapi', 'fred', 'secEdgar', 'prompt_chaining', 'evaluator']. "
            "The 'parameters' object should contain the necessary arguments for the tool. "
            "For example, a step could be: {\\\\\\\"tool\\\\\\\": \\\\\\\"newsapi\\\\\\\", \\\\\\\"parameters\\\\\\\": {\\\\\\\"symbol\\\\\\\": \\\\\\\"NVDA\\\\\\\"}}. "
            "Generate a plan of 5-7 critical steps to form an investment thesis."
        )
        
        user_prompt = self.prompt_manager.get_prompt('planning_prompt', symbol=symbol, memory=memory)
            
        response = call_gemini(system_instruction, user_prompt, json_output=True)
        
        if response and isinstance(response, list):
            return response
        else:
            logger.error("Failed to generate a valid plan.")
            return []
