import os
import logging
from utils.llm_integration import call_gemini, call_judge_gemini
from utils.prompt_manager import PromptManager

class EvaluatorOptimizerAgent:
    def __init__(self):
        self.prompt_manager = PromptManager()

    def run(self, data: dict, symbol: str) -> str:
        """Runs the evaluator-optimizer workflow to generate a polished investment thesis."""
        
        # 1. Optimizer (Drafting)
        draft_prompt = self.prompt_manager.get_prompt('draft_thesis', data=data, symbol=symbol)
        draft = call_gemini("You are a financial analyst drafting an investment thesis.", draft_prompt, json_output=False)
        
        if not draft:
            return "Failed to generate a draft."

        logging.info("\n--- Initial Draft ---")
        logging.info(draft)

        # 2. Evaluator (Critique)
        evaluator_prompt = self.prompt_manager.get_prompt('evaluate_thesis', draft=draft, symbol=symbol)
        critique = call_judge_gemini("You are a meticulous financial evaluator.", evaluator_prompt, json_output=False)

        if not critique:
            return "Failed to generate a critique."

        logging.info("\n--- Critique ---")
        logging.info(critique)

        # 3. Optimizer (Refinement)
        refinement_prompt = self.prompt_manager.get_prompt('refine_thesis', draft=draft, critique=critique, symbol=symbol)
        final_thesis = call_gemini("You are a financial analyst refining your work.", refinement_prompt, json_output=False)

        if not final_thesis:
            return "Failed to generate the final thesis."

        logging.info("\n--- Final Thesis ---")
        logging.info(final_thesis)

        return final_thesis
