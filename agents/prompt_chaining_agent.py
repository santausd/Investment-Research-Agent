import os
from utils.llm_integration import call_gemini
from utils.prompt_manager import PromptManager
from utils.logger_config import logger

class PromptChainingAgent:
    def __init__(self):
        self.prompt_manager = PromptManager()

    def run(self, raw_text: str) -> dict:
        """Runs a 5-stage prompt chain to process raw text."""
        
        # Stage 1: Ingest/Preprocess
        preprocess_prompt = self.prompt_manager.get_prompt('preprocess_text', text=raw_text)
        clean_text = call_gemini("You are a text cleaning assistant.", preprocess_prompt, json_output=False)
        if not clean_text:
            logger.error("Failed to clean text.")
            return {"error": "Failed to clean text."}
        logger.info("--- Cleaned Text ---")
        logger.debug(clean_text)

        # Stage 2: Classify
        classify_prompt = self.prompt_manager.get_prompt('classify_document', text=clean_text)
        classification = call_gemini("You are a text classification specialist.", classify_prompt, json_output=False)
        if not classification:
            logger.error("Failed to classify text.")
            return {"error": "Failed to classify text."}
        logger.info(f"--- Classification ---\n{classification}")

        # Stage 3: Extract
        extract_prompt = self.prompt_manager.get_prompt('extract_key_info', text=clean_text)
        extracted_data = call_gemini("You are a data extraction expert.", extract_prompt, json_output=True)
        if not extracted_data:
            logger.error("Failed to extract data.")
            return {"error": "Failed to extract data."}
        logger.info(f"--- Extracted Data ---")
        logger.debug(extracted_data)

        # Stage 4: Summarize
        summarize_prompt = self.prompt_manager.get_prompt('summarize_news', article=clean_text)
        summary = call_gemini("You are a financial news summarizer.", summarize_prompt, json_output=False)
        if not summary:
            logger.error("Failed to summarize text.")
            return {"error": "Failed to summarize text."}
        logger.info(f"--- Summary ---\n{summary}")

        return {
            "classification": classification.strip(),
            "extracted_data": extracted_data,
            "summary": summary.strip()
        }
