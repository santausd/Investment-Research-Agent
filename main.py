import json
import os
from agents.toolbox_agent import ToolboxAgent
from agents.planning_agent import PlanningAgent
from agents.memory_agent import MemoryAgent
from agents.evaluator_optimizer_agent import EvaluatorOptimizerAgent
from agents.prompt_chaining_agent import PromptChainingAgent
from agents.routing_agent import RoutingAgent
from utils.utils import load_env

def run_analysis(symbol: str):
    """Runs the full agentic analysis for a given stock symbol."""
    
    # Load API keys 
    load_env()

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
        if "yfinance" in step.lower():
            state["raw_data"]['yfinance'] = toolbox.fetch('yfinance', symbol)
        elif "news" in step.lower():
            state["raw_data"]['news'] = toolbox.fetch('newsapi', symbol)
        elif "economic" in step.lower() or "fred" in step.lower():
            # A more robust implementation would parse the indicator
            state["raw_data"]['fred_gdp'] = toolbox.fetch('fred', 'GDP')

    print("\n--- Fetched Raw Data ---")
    # Abridged printing for brevity
    if 'yfinance' in state["raw_data"]:
        print("  - Yahoo Finance data retrieved.")
    if 'news' in state["raw_data"]:
        print("  - News data retrieved.")
    if 'fred_gdp' in state["raw_data"]:
        print("  - FRED GDP data retrieved.")

    # Toolbox Output -> Prompt Chaining Agent -> Routing Agent
    if 'news' in state["raw_data"] and state["raw_data"]['news']['articles']:
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
