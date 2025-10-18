import os
import numpy as np
import ollama
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

class MultiAgentEvaluator:
    def __init__(self):
        self.openai_model = "gpt-4o"
        self.ollama_model = "llama2"
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

        # Initialize OpenAI client only if key is set
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key and OpenAI is not None:
            try:
                self.client = OpenAI(api_key=api_key)
                self.mode = "openai"
            except Exception:
                self.client = None
                self.mode = "ollama"
        else:
            self.client = None
            self.mode = "ollama"

        print(f"Evaluator initialized in {self.mode.upper()} mode")

    def llm_grade(self, thesis: str, reference: str = None) -> dict:
        """Evaluate investment thesis quality using OpenAI or Ollama."""
        prompt = f"""
        Evaluate this investment thesis for clarity, factual accuracy, and rigor.
        Rate each dimension from 1–10 and summarize with justification.

        Thesis:
        {thesis}

        Reference (if provided):
        {reference}
        """

        # --- Try OpenAI first ---
        if self.mode == "openai" and self.client is not None:
            try:
                response = self.client.chat.completions.create(
                    model=self.openai_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                )
                return {"source": "openai", "raw": response.choices[0].message.content}
            except Exception as e:
                print(f"[OpenAI Error] {e} — Falling back to Ollama.")
                self.mode = "ollama"

        # --- Fallback to Ollama ---
        if ollama is None:
            return {"error": "Neither OpenAI nor Ollama available."}

        try:
            response = ollama.chat(
                model=self.ollama_model,
                messages=[{"role": "user", "content": prompt}],
            )
            return {"source": "ollama", "raw": response["message"]["content"]}
        except Exception as e:
            return {"error": f"Both evaluators failed: {e}"}

    def embedding_consistency(self, thesis_a: str, thesis_b: str) -> float:
        """Measure semantic similarity between two analyses."""
        embeddings = self.embedder.encode([thesis_a, thesis_b])
        return cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

    def coordination_efficiency(self, logs: list) -> dict:
        """Analyze inter-agent message structure."""
        n_messages = len(logs)
        avg_message_len = np.mean([len(m["content"]) for m in logs])
        return {"n_messages": n_messages, "avg_message_len": avg_message_len}

