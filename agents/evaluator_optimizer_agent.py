import os
from utils.llm_integration import call_gemini

class EvaluatorOptimizerAgent:
    def __init__(self):
        pass

    def run(self, data: dict) -> str:
        """Runs the evaluator-optimizer workflow to generate a polished investment thesis."""
        
        # 1. Optimizer (Drafting)
        draft_prompt = "Generate a comprehensive draft investment analysis and thesis (Buy/Hold/Sell) based on the following data." 
        draft_prompt += f"\n\nData:\n{data}"
        draft = call_gemini("You are a financial analyst drafting an investment thesis.", draft_prompt, json_output=False)
        
        if not draft:
            return "Failed to generate a draft."

        print("\n--- Initial Draft ---")
        print(draft)

        # 2. Evaluator (Critique)
        evaluator_prompt = (
            "Critique the following investment draft for two things: "
            "1. Factual consistency (do the numbers match the source data?) "
            "2. Logical consistency (is the 'Buy' recommendation justified by the identified risks?). "
            "Provide a specific suggestion for refinement."
        )
        evaluator_prompt += f"\n\nDraft:\n{draft}"
        critique = call_gemini("You are a meticulous financial evaluator.", evaluator_prompt, json_output=False)

        if not critique:
            return "Failed to generate a critique."

        print("\n--- Critique ---")
        print(critique)

        # 3. Optimizer (Refinement)
        refinement_prompt = (
            "Based on the critique provided, refine and correct the initial draft. "
            "Produce the final, polished investment thesis."
        )
        refinement_prompt += f"\n\nInitial Draft:\n{draft}\n\nCritique:\n{critique}"
        final_thesis = call_gemini("You are a financial analyst refining your work.", refinement_prompt, json_output=False)

        if not final_thesis:
            return "Failed to generate the final thesis."

        print("\n--- Final Thesis ---")
        print(final_thesis)

        return final_thesis
