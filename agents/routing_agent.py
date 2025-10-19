from utils.llm_integration import call_gemini
from utils.prompt_manager import PromptManager

class RoutingAgent:
    def __init__(self):
        self.prompt_manager = PromptManager()

    def route(self, classification: str) -> str:
        """DetermTines the next step based on the classification using an LLM."""
        
        prompt = self.prompt_manager.get_prompt('routing_prompt', classification=classification)
        
        # We don't expect a JSON output, so we set json_output=False
        model_name = call_gemini("You are a routing agent.", prompt, json_output=False)
        
        if model_name and model_name in ['EarningsModelRun', 'ComplianceCheck', 'MarketImpactAnalysis', 'GeneralAnalysis']:
            return model_name
        else:
            # Default to GeneralAnalysis if the model returns an invalid or empty response
            return 'GeneralAnalysis'
