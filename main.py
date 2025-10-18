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
from utils.logger_config import logger

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

    logger.info(f"--- Starting Analysis for {symbol} ---")

    # 3. Establish Flow
    # Input Symbol -> Memory Agent -> Planning Engine Agent
    retrieved_memory = memory.retrieve(symbol)
    if retrieved_memory:
        logger.info(f"--- Retrieved Memory for {symbol} ---")
        logger.debug(json.dumps(retrieved_memory, indent=4))
    
    state["plan"] = planner.generate_plan(symbol, json.dumps(retrieved_memory) if retrieved_memory else None)
    if not state["plan"]:
        logger.error("Could not generate a plan. Exiting.")
        return

    logger.info(f"--- Generated Plan for {symbol} ---")
    logger.debug(json.dumps(state["plan"], indent=4))

    # 4. Execute the structured plan
    logger.info("--- Executing Plan ---")
    for step in state["plan"]:
        tool = step.get("tool")
        params = step.get("parameters", {})
        
        if not tool:
            logger.warning(f"Skipping invalid step: {step}")
            continue

        logger.info(f"--> Executing: {tool} with params: {params}")

        result = None
        if tool in ['yfinance', 'newsapi', 'fred', 'secEdgar']:
            # Add symbol to params if not present for relevant tools
            if tool in ['yfinance', 'newsapi', 'secEdgar'] and 'symbol' not in params:
                params['symbol'] = symbol
            
            result = toolbox.fetch(tool, **params)
            state["raw_data"][tool] = result
            if result:
                logger.info(f"  - SUCCESS: Data retrieved with tool '{tool}'.")
            else:
                logger.error(f"  - FAILED: Tool '{tool}' did not return data.")
        
        elif tool == 'prompt_chaining':
            # This tool processes news articles. We need to find them in raw_data.
            news_data = state["raw_data"].get('newsapi')
            if news_data and news_data.get('articles'):
                for article in news_data['articles']:
                    processed_article = prompt_chainer.run(article['title'] + "\n" + article.get('description', ''))
                    state["processed_news"].append(processed_article)
                    
                    # The routing logic is coupled with prompt chaining for now
                    route = router.route(processed_article.get('classification', ''))
                    logger.info(f"  --- Routing for article: \"{article['title']}\" ---")
                    logger.debug(f"    - Classification: {processed_article.get('classification')}")
                    logger.debug(f"    - Route: {route}")
                    # Placeholder execution
                    if route == 'EarningsModelRun':
                        logger.info("    - (Placeholder) Would run a discounted cash flow model here.")
                    elif route == 'ComplianceCheck':
                        logger.info("    - (Placeholder) Would run a regulatory impact model here.")
                    else:
                        logger.info("    - (Placeholder) Would run a general analysis model here.")
            else:
                logger.warning("  - SKIPPING: News data not available for prompt_chaining.")

        elif tool == 'evaluator':
            logger.info("  --- Generating Final Thesis with Evaluator-Optimizer ---")
            result = evaluator.run(state, state['symbol'])
            state["final_thesis"] = result
        
        else:
            logger.warning(f"  - WARNING: Tool '{tool}' not recognized in execution loop.")

    # 5. Update Memory and Final Output
    logger.info("--- Finalizing Analysis ---")
    if state["final_thesis"]:
        memory.update(symbol, state)

    logger.warning(f"--- Completed Analysis for {symbol} ---")
    logger.warning("Final Thesis:")
    logger.warning(state["final_thesis"])

if __name__ == '__main__':
    #agent = ToolboxAgent()
    #symbol = "NVDA"
    #result = agent.get_yahoo_finance_data(symbol)
    #print (result)
    run_analysis('NVDA')
