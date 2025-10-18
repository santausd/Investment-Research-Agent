from utils.llm_integration import call_gemini
from utils.prompt_manager import PromptManager

class RoutingAgent:
    def __init__(self):
        self.prompt_manager = PromptManager()

    def route(self, classification: str) -> str:
        """Determines the next step based on the classification using an LLM."""
        
        prompt = self.prompt_manager.get_prompt('routing_prompt', classification=classification)
        
        # Using a generic system prompt as the main instruction is in the user prompt
        system_prompt = "You are an intelligent routing agent."
        
        model_name = call_gemini(system_prompt, prompt, json_output=False)
        
        # Basic validation to ensure the model returns a valid choice
        valid_models = ['EarningsModelRun', 'ComplianceCheck', 'MarketImpactAnalysis', 'GeneralAnalysis']
        if model_name and model_name.strip() in valid_models:
            return model_name.strip()
        else:
            # Fallback to a default model if the LLM fails or returns an invalid response
            return 'GeneralAnalysis'
