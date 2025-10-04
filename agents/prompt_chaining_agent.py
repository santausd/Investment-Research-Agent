import os
from utils.llm_integration import call_gemini

class PromptChainingAgent:
    def __init__(self):
        pass

    def run(self, raw_text: str) -> dict:
        """Runs a 5-stage prompt chain to process raw text."""
        
        # Stage 1: Ingest/Preprocess
        preprocess_prompt = f"Clean the following text and remove any boilerplate content:\n\n{raw_text}"
        clean_text = call_gemini("You are a text cleaning assistant.", preprocess_prompt, json_output=False)
        if not clean_text:
            return {"error": "Failed to clean text."}
        print("\n--- Cleaned Text ---")
        print(clean_text)

        # Stage 2: Classify
        classify_prompt = f"What is the primary event type in this text? (e.g., Earnings, Product Launch, Regulation, Macro):\n\n{clean_text}"
        classification = call_gemini("You are a text classification specialist.", classify_prompt, json_output=False)
        if not classification:
            return {"error": "Failed to classify text."}
        print(f"\n--- Classification ---\n{classification}")

        # Stage 3: Extract
        extract_prompt = f"Extract all numerical data points (e.g., EPS, Revenue, Guidance) mentioned in the text:\n\n{clean_text}"
        extracted_data = call_gemini("You are a data extraction expert.", extract_prompt, json_output=True)
        if not extracted_data:
            return {"error": "Failed to extract data."}
        print(f"\n--- Extracted Data ---\n{extracted_data}")

        # Stage 4: Summarize
        summarize_prompt = f"Write a concise, abstractive summary of the key market takeaway (1-2 sentences):\n\n{clean_text}"
        summary = call_gemini("You are a financial news summarizer.", summarize_prompt, json_output=False)
        if not summary:
            return {"error": "Failed to summarize text."}
        print(f"\n--- Summary ---\n{summary}")

        return {
            "classification": classification.strip(),
            "extracted_data": extracted_data,
            "summary": summary.strip()
        }
