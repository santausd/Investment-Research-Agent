import traceback
from utils.llm_integration import call_gemini
from utils.logger import AgentLogger

class PromptChainingAgent:
    def __init__(self):
        pass

    def _get_logger(self, state):
        """Attach logger to agent if available."""
        return AgentLogger(state) if state and "conversation_logs" in state else None

    def run(self, raw_text: str, state: dict = None) -> dict:
        """Runs a 5-stage prompt chain to process raw text with detailed logging."""

        logger = self._get_logger(state)
        results = {}

        try:
            # --------------------------------------------------------------------------------
            # Stage 1: Ingest / Preprocess
            # --------------------------------------------------------------------------------
            preprocess_prompt = f"Clean the following text and remove any boilerplate content:\n\n{raw_text}"
            if logger:
                logger.log("PromptChainingAgent", "System", "Stage 1: Preprocessing text input.")
            clean_text = call_gemini("You are a text cleaning assistant.", preprocess_prompt, json_output=False)

            if not clean_text:
                msg = "Failed to clean text."
                if logger:
                    logger.log("PromptChainingAgent", "System", msg, level="error")
                return {"error": msg}

            if logger:
                logger.log("PromptChainingAgent", "System", "Text cleaned successfully.", payload={"clean_text": clean_text[:500]})
            print("\n--- Cleaned Text ---")
            print(clean_text)

            # --------------------------------------------------------------------------------
            # Stage 2: Classification
            # --------------------------------------------------------------------------------
            classify_prompt = f"What is the primary event type in this text? (e.g., Earnings, Product Launch, Regulation, Macro):\n\n{clean_text}"
            if logger:
                logger.log("PromptChainingAgent", "System", "Stage 2: Classifying text.")
            classification = call_gemini("You are a text classification specialist.", classify_prompt, json_output=False)

            if not classification:
                msg = "Failed to classify text."
                if logger:
                    logger.log("PromptChainingAgent", "System", msg, level="error")
                return {"error": msg}

            if logger:
                logger.log("PromptChainingAgent", "System", "Classification complete.", payload={"classification": classification.strip()})
            print(f"\n--- Classification ---\n{classification}")
            results["classification"] = classification.strip()

            # --------------------------------------------------------------------------------
            # Stage 3: Extraction
            # --------------------------------------------------------------------------------
            extract_prompt = f"Extract all numerical data points (e.g., EPS, Revenue, Guidance) mentioned in the text:\n\n{clean_text}"
            if logger:
                logger.log("PromptChainingAgent", "System", "Stage 3: Extracting numerical data.")
            extracted_data = call_gemini("You are a data extraction expert.", extract_prompt, json_output=True)

            if not extracted_data:
                msg = "Failed to extract data."
                if logger:
                    logger.log("PromptChainingAgent", "System", msg, level="error")
                return {"error": msg}

            if logger:
                logger.log("PromptChainingAgent", "System", "Data extraction complete.", payload={"extracted_data": extracted_data})
            print(f"\n--- Extracted Data ---\n{extracted_data}")
            results["extracted_data"] = extracted_data

            # --------------------------------------------------------------------------------
            # Stage 4: Summarization
            # --------------------------------------------------------------------------------
            summarize_prompt = f"Write a concise, abstractive summary of the key market takeaway (1-2 sentences):\n\n{clean_text}"
            if logger:
                logger.log("PromptChainingAgent", "System", "Stage 4: Summarizing content.")
            summary = call_gemini("You are a financial news summarizer.", summarize_prompt, json_output=False)

            if not summary:
                msg = "Failed to summarize text."
                if logger:
                    logger.log("PromptChainingAgent", "System", msg, level="error")
                return {"error": msg}

            if logger:
                logger.log("PromptChainingAgent", "System", "Summary complete.", payload={"summary": summary.strip()})
            print(f"\n--- Summary ---\n{summary}")
            results["summary"] = summary.strip()

            # --------------------------------------------------------------------------------
            # Final Results
            # --------------------------------------------------------------------------------
            if logger:
                logger.log("PromptChainingAgent", "System", "Prompt chaining complete.", payload=results)
            return results

        except Exception as e:
            error_details = traceback.format_exc()
            if logger:
                logger.log("PromptChainingAgent", "System",
                           f"Unhandled exception in prompt chain: {e}",
                           level="error",
                           traceback=error_details)
            return {"error": f"Unhandled exception: {e}"}

