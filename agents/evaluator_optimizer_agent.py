import traceback
from utils.llm_integration import call_gemini
from utils.logger import AgentLogger

class EvaluatorOptimizerAgent:
    def __init__(self):
        pass

    def _get_logger(self, state):
        """Attach logger to agent if state has conversation logs."""
        return AgentLogger(state) if state and "conversation_logs" in state else None

    def run(self, data: dict, state: dict = None) -> str:
        """Runs the evaluator-optimizer workflow with detailed logging."""

        logger = self._get_logger(state)
        try:
            # --------------------------------------------------------------------------------
            # 1. Optimizer Stage — Draft Thesis
            # --------------------------------------------------------------------------------
            draft_prompt = (
                "Generate a comprehensive draft investment analysis and thesis "
                "(Buy/Hold/Sell) based on the following data.\n\nData:\n"
                f"{data}"
            )
            if logger:
                logger.log("EvaluatorOptimizerAgent", "System", "Stage 1: Generating initial draft thesis.")
            draft = call_gemini("You are a financial analyst drafting an investment thesis.", draft_prompt, json_output=False)

            if not draft:
                msg = "Failed to generate a draft."
                if logger:
                    logger.log("EvaluatorOptimizerAgent", "System", msg, level="error")
                return msg

            if logger:
                logger.log("EvaluatorOptimizerAgent", "System", "Draft thesis generated successfully.",
                           payload={"draft": draft[:500]})
            print("\n--- Initial Draft ---")
            print(draft)

            # --------------------------------------------------------------------------------
            # 2. Evaluator Stage — Critique Draft
            # --------------------------------------------------------------------------------
            evaluator_prompt = (
                "Critique the following investment draft for two things:\n"
                "1. Factual consistency (do the numbers match the source data?)\n"
                "2. Logical consistency (is the 'Buy' recommendation justified by the identified risks?).\n"
                "Provide a specific suggestion for refinement.\n\n"
                f"Draft:\n{draft}"
            )
            if logger:
                logger.log("EvaluatorOptimizerAgent", "System", "Stage 2: Evaluating draft for consistency and logic.")
            critique = call_gemini("You are a meticulous financial evaluator.", evaluator_prompt, json_output=False)

            if not critique:
                msg = "Failed to generate a critique."
                if logger:
                    logger.log("EvaluatorOptimizerAgent", "System", msg, level="error")
                return msg

            if logger:
                logger.log("EvaluatorOptimizerAgent", "System", "Critique generated successfully.",
                           payload={"critique": critique[:500]})
            print("\n--- Critique ---")
            print(critique)

            # --------------------------------------------------------------------------------
            # 3. Optimizer Stage — Refinement
            # --------------------------------------------------------------------------------
            refinement_prompt = (
                "Based on the critique provided, refine and correct the initial draft. "
                "Produce the final, polished investment thesis.\n\n"
                f"Initial Draft:\n{draft}\n\nCritique:\n{critique}"
            )
            if logger:
                logger.log("EvaluatorOptimizerAgent", "System", "Stage 3: Refining draft based on critique.")
            final_thesis = call_gemini("You are a financial analyst refining your work.", refinement_prompt, json_output=False)

            if not final_thesis:
                msg = "Failed to generate the final thesis."
                if logger:
                    logger.log("EvaluatorOptimizerAgent", "System", msg, level="error")
                return msg

            if logger:
                logger.log("EvaluatorOptimizerAgent", "System", "Final polished thesis generated successfully.",
                           payload={"final_thesis": final_thesis[:500]})
            print("\n--- Final Thesis ---")
            print(final_thesis)

            return final_thesis

        except Exception as e:
            error_details = traceback.format_exc()
            if logger:
                logger.log("EvaluatorOptimizerAgent", "System",
                           f"Unhandled exception in evaluator-optimizer pipeline: {e}",
                           level="error",
                           traceback=error_details)
            return f"Unhandled exception: {e}"

