import json
import os
import re
import google.generativeai as genai
from agents.toolbox_agent import ToolboxAgent
from agents.planning_agent import PlanningAgent
from agents.memory_agent import MemoryAgent
from agents.evaluator_optimizer_agent import EvaluatorOptimizerAgent
from agents.prompt_chaining_agent import PromptChainingAgent
from agents.routing_agent import RoutingAgent
from evaluation.evaluator import MultiAgentEvaluator
from utils.utils import load_env

def run_analysis(symbol: str):
    """Runs the full agentic analysis for a given stock symbol."""
    
    # Load API keys and configure Gemini
    load_env()
    genai.configure(api_key=os.environ.get('GOOGLE_API_KEY'))

    # 1. Initialize Agents
    toolbox = ToolboxAgent()
    memory = MemoryAgent()
    planner = PlanningAgent()
    prompt_chainer = PromptChainingAgent()
    router = RoutingAgent()
    evaluator = EvaluatorOptimizerAgent()

    # 2. Define State
    state = {
        "symbol": symbol,
        "plan": [],
        "raw_data": {},
        "processed_news": [],
        "conversational_logs": [],
        "final_thesis": None
    }

    print(f"--- Starting Analysis for {symbol} ---")

    # 3. Establish Flow
    # Input Symbol -> Memory Agent -> Planning Engine Agent
    retrieved_memory = memory.retrieve(symbol, state)
    if retrieved_memory:
        print(f"\n--- Retrieved Memory for {symbol} ---")
        print(json.dumps(retrieved_memory, indent=4))
    
    state["plan"] = planner.generate_plan(symbol, state, json.dumps(retrieved_memory) if retrieved_memory else None)
    if not state["plan"]:
        print("Could not generate a plan. Exiting.")
        return

    print(f"\n--- Generated Plan for {symbol} ---")
    for step in state["plan"]:
        print(f"- {step}")

    # The sequence then calls the Toolbox Agent multiple times
    for step in state["plan"]:
        if ("assessment" in step.lower() or "analysis" in step.lower()) and 'yfinance' not in state["raw_data"]:
            state["raw_data"]['yfinance'] = toolbox.fetch('yfinance', symbol, state)
        if ("news" in step.lower() or "finding" in step.lower() or "analysis" in step.lower()) and 'news' not in state["raw_data"]:
            state["raw_data"]['news'] = toolbox.fetch('newsapi', symbol, state)
        if ("economic" in step.lower() or "advancements" in step.lower()) and 'fred_gdp' not in state["raw_data"]:
            # A more robust implementation would parse the indicator
            state["raw_data"]['fred_gdp'] = toolbox.fetch('fred', 'GDP', state)
        if ("valuation" in step.lower() or "risk" in step.lower() or "report" in step.lower()) and "secEdgar" not in state["raw_data"]:
            # A more robust implementation would parse the indicator
            state["raw_data"]['secEdgar'] = toolbox.fetch('secEdgar', symbol, state)

    print("\n--- Fetched Raw Data ---")
    # Abridged printing for brevity
    if 'yfinance' in state["raw_data"]:
        print("  - Yahoo Finance data retrieved.")
    if 'news' in state["raw_data"]:
        print("  - News data retrieved.")
    if 'fred_gdp' in state["raw_data"]:
        print("  - FRED GDP data retrieved.")
    if 'secEdgar' in state["raw_data"]:
        print("  - Sec Edgar data retrieved.")

    # Toolbox Output -> Prompt Chaining Agent -> Routing Agent
    if 'news' in state["raw_data"] and state["raw_data"]['news']!=None and state["raw_data"]['news']['articles']:
        for article in state["raw_data"]['news']['articles']:
            processed_article = prompt_chainer.run(article['title'] + "\n" + article.get('description', ''), state)
            state["processed_news"].append(processed_article)
            
            state["classification"] = router.route(processed_article.get('classification', ''), state)

            print(f"\n--- Routing for article: '{article['title']}' ---")
            print(f"  - Classification: {processed_article.get('classification')}")
            print(f"  - Route: {state["classification"]}")

            # Routing -> Execution of Specialized Model (Placeholder)
            if state["classification"] == 'EarningsModelRun':
                print("  - (Placeholder) Would run a discounted cash flow model here.")
            elif state["classification"] == 'ComplianceCheck':
                print("  - (Placeholder) Would run a regulatory impact model here.")
            else:
                print("  - (Placeholder) Would run a general analysis model here.")

    # All data -> Evaluator–Optimizer Agent
    print("\n--- Generating Final Thesis with Evaluator-Optimizer ---")

    # Ensure yfinance data exists and is a non-empty list
    #yfinance_data = state.get("raw_data", {}).get("yfinance", [])

    #if yfinance_data and isinstance(yfinance_data, list):
    #    financial_data = yfinance_data[0]  # first (and usually only) entry
    #else:
    #    financial_data = {}

    # Collect all relevant structured data for evaluation
    evaluator_data = {
        "symbol": state.get("symbol"),
        "classification": state.get("classification"),
        "financials": state.get("raw_data", {}).get("yfinance", []),
        "news": state.get("processed_news"),
        "economics": state.get("raw_data", {}).get("fred_gdp", []),
        "filings": state.get("raw_data", {}).get("secEdgar", [])
    }

    state["final_thesis"] = evaluator.run(evaluator_data, state)

    final_thesis = state["final_thesis"]
    logs = state.get("conversation_logs", [])

    evaluator = MultiAgentEvaluator()

    # LLM-based evaluation
    eval_result = evaluator.llm_grade(final_thesis)
    # Coordination metrics
    coordination = evaluator.coordination_efficiency(logs)

    # Basic heuristic parsing of scores from LLM text
    eval_text = eval_result.get("raw", "")
    clarity = factual = rigor = overall = 0

    # Regex patterns to extract scores
    patterns = {
        "clarity": r"clarity[:\s]*([0-9]+)\s*/\s*10",
        "accuracy": r"accuracy[:\s]*([0-9]+)\s*/\s*10",
        "rigor": r"rigor[:\s]*([0-9]+)\s*/\s*10",
        "overall": r"overall.*?([0-9]+)\s*/\s*10"
    }
    
    # Extract scores
    for key, pattern in patterns.items():
        match = re.search(pattern, eval_text, re.IGNORECASE)
        if match:
            score = int(match.group(1))
            if key == "clarity":
                clarity = score
            elif key == "accuracy":
                accuracy = score
            elif key == "rigor":
                rigor = score
            elif key == "overall":
                overall = score

    eval_metrics = {
        "clarity": clarity,
        "accuracy": accuracy,
        "rigor": rigor,
        "overall": overall,
        "source": eval_result.get("source", "unknown"),
        "evaluation_summary": eval_text,
    }

    state["evaluation"] = eval_metrics

    print("\n--- Evaluation Metrics ---")
    print(eval_metrics)

    memory.update(symbol, state["evaluation"])

    # Evaluator–Optimizer Output -> Memory Agent (Update)
    if state["final_thesis"]:
        # A more robust implementation would extract key metrics from the thesis
        memory.update(symbol, {"summary": state["final_thesis"]}, state)

        print(f"\n--- Completed Analysis for {symbol} ---")
        print("Final Thesis:")
        print(state["final_thesis"])
    
        return state


    return f"No summary generated for the symbol: {symbol}"

	
if __name__ == "__main__":
    # Prompt user for input with default value
    user_input = input("Enter stock symbol [default: NVDA]: ").strip()
    
    # Use NVDA if no input is provided
    symbol = user_input.upper() if user_input else "NVDA"
    
    run_analysis(symbol)
