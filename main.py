import json
import os
import google.generativeai as genai
from agents.toolbox_agent import ToolboxAgent
from agents.planning_agent import PlanningAgent
from agents.memory_agent import MemoryAgent
from agents.evaluator_optimizer_agent import EvaluatorOptimizerAgent
from agents.prompt_chaining_agent import PromptChainingAgent
from agents.routing_agent import RoutingAgent
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
        "final_thesis": None
    }

    print(f"--- Starting Analysis for {symbol} ---")

    # 3. Establish Flow
    # Input Symbol -> Memory Agent -> Planning Engine Agent
    retrieved_memory = memory.retrieve(symbol)
    if retrieved_memory:
        print(f"\n--- Retrieved Memory for {symbol} ---")
        print(json.dumps(retrieved_memory, indent=4))
    
    state["plan"] = planner.generate_plan(symbol, json.dumps(retrieved_memory) if retrieved_memory else None)
    if not state["plan"]:
        print("Could not generate a plan. Exiting.")
        return

    print(f"\n--- Generated Plan for {symbol} ---")
    for step in state["plan"]:
        print(f"- {step}")

    # The sequence then calls the Toolbox Agent multiple times
    for step in state["plan"]:
        if ("assessment" in step.lower() or "analysis" in step.lower()) and 'yfinance' not in state["raw_data"]:
            state["raw_data"]['yfinance'] = toolbox.fetch('yfinance', symbol)
        if ("news" in step.lower() or "finding" in step.lower() or "analysis" in step.lower()) and 'news' not in state["raw_data"]:
            state["raw_data"]['news'] = toolbox.fetch('newsapi', symbol)
        if ("economic" in step.lower() or "advancements" in step.lower()) and 'fred_gdp' not in state["raw_data"]:
            # A more robust implementation would parse the indicator
            state["raw_data"]['fred_gdp'] = toolbox.fetch('fred', 'GDP')
        if ("valuation" in step.lower() or "risk" in step.lower() or "report" in step.lower()) and "secEdgar" not in state["raw_data"]:
            # A more robust implementation would parse the indicator
            state["raw_data"]['secEdgar'] = toolbox.fetch('secEdgar', symbol)

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
            processed_article = prompt_chainer.run(article['title'] + "\n" + article.get('description', ''))
            state["processed_news"].append(processed_article)
            
            route = router.route(processed_article.get('classification', ''))
            print(f"\n--- Routing for article: '{article['title']}' ---")
            print(f"  - Classification: {processed_article.get('classification')}")
            print(f"  - Route: {route}")

            # Routing -> Execution of Specialized Model (Placeholder)
            if route == 'EarningsModelRun':
                print("  - (Placeholder) Would run a discounted cash flow model here.")
            elif route == 'ComplianceCheck':
                print("  - (Placeholder) Would run a regulatory impact model here.")
            else:
                print("  - (Placeholder) Would run a general analysis model here.")

    # All data -> Evaluator–Optimizer Agent
    print("\n--- Generating Final Thesis with Evaluator-Optimizer ---")
    state["final_thesis"] = evaluator.run(state)

    # Evaluator–Optimizer Output -> Memory Agent (Update)
    if state["final_thesis"]:
        # A more robust implementation would extract key metrics from the thesis
        memory.update(symbol, {"summary": state["final_thesis"]})

    print(f"\n--- Completed Analysis for {symbol} ---")
    print("Final Thesis:")
    print(state["final_thesis"])

if __name__ == '__main__':
    run_analysis('NVDA')
