import json
import os
import argparse
import logging
import google.generativeai as genai
from agents.toolbox_agent import ToolboxAgent
from agents.planning_agent import PlanningAgent
from agents.memory_agent import MemoryAgent
from agents.evaluator_optimizer_agent import EvaluatorOptimizerAgent
from agents.prompt_chaining_agent import PromptChainingAgent
from agents.routing_agent import RoutingAgent
from utils.utils import load_env, setup_logging

class InvestmentAnalysisOrchestrator:
    """
    Orchestrates the investment analysis process by coordinating various agents.
    """
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.state = {
            "symbol": self.symbol,
            "plan": [],
            "raw_data": {},
            "processed_news": [],
            "final_thesis": None
        }
        self._initialize_agents()

    def _initialize_agents(self):
        """Initializes all the agents required for the analysis."""
        self.toolbox = ToolboxAgent()
        self.memory = MemoryAgent()
        self.planner = PlanningAgent()
        self.prompt_chainer = PromptChainingAgent()
        self.router = RoutingAgent()
        self.evaluator = EvaluatorOptimizerAgent()

        self.tool_executors = {
            'yfinance': self.toolbox.fetch,
            'newsapi': self.toolbox.fetch,
            'fred': self.toolbox.fetch,
            'secEdgar': self.toolbox.fetch,
            'query_sec_filings': self.toolbox.fetch,
            'prompt_chaining': self._process_news,
            'evaluator': self._run_evaluator
        }
        logging.info("All agents initialized.")

    def _generate_plan(self):
        """Generates an analysis plan using the planning agent."""
        logging.info(f"Generating plan for {self.symbol}...")
        retrieved_memory = self.memory.retrieve(self.symbol)
        if retrieved_memory:
            logging.info(f"Retrieved memory for {self.symbol}.")
            logging.info(json.dumps(retrieved_memory, indent=4))

        self.state["plan"] = self.planner.generate_plan(
            self.symbol, retrieved_memory
        )
        if not self.state["plan"]:
            logging.error("Could not generate a plan. Exiting.")
            return False
        
        logging.info(f"Generated plan for {self.symbol}:")
        logging.info(json.dumps(self.state["plan"], indent=4))
        return True

    def _execute_plan(self):
        """Executes the generated plan step by step."""
        logging.info("Executing plan...")
        for step in self.state["plan"]:
            tool = step.get("tool")
            params = step.get("parameters", {})
            
            if not tool:
                logging.warning(f"Skipping invalid step: {step}")
                continue

            logging.info(f"Executing tool: {tool} with params: {params}")
            
            executor = self.tool_executors.get(tool)
            if executor:
                try:
                    if tool in ['yfinance', 'newsapi', 'fred', 'secEdgar', 'query_sec_filings']:
                        if tool in ['yfinance', 'newsapi', 'secEdgar', 'query_sec_filings'] and 'symbol' not in params:
                            params['symbol'] = self.symbol
                        result = executor(tool_name=tool, **params)
                        self.state["raw_data"][tool] = result
                        if result:
                            logging.info(f"SUCCESS: Data retrieved with tool '{tool}'.")
                        else:
                            logging.warning(f"FAILED: Tool '{tool}' did not return data.")
                    else:
                        executor()
                except Exception as e:
                    logging.error(f"An error occurred during the execution of tool '{tool}': {e}", exc_info=True)
            else:
                logging.warning(f"Tool '{tool}' not recognized in execution loop.")

    def _process_news(self):
        """Processes news articles using the prompt chaining and routing agents."""
        logging.info("Processing news articles...")
        news_data = self.state["raw_data"].get('newsapi')
        if news_data and news_data.get('articles'):
            for article in news_data['articles']:
                processed_article = self.prompt_chainer.run(
                    article['title'] + "\n" + article.get('description', '')
                )
                self.state["processed_news"].append(processed_article)
                
                # Routing logic
                classification = processed_article.get('classification', '')
                route = self.router.route(classification)
                logging.info(f"Article '{article['title']}' classified as '{classification}', routed to '{route}'.")
                # Placeholder for executing routed models
                self._execute_routed_model(route)
        else:
            logging.info("Skipping news processing: No news data available.")

    def _execute_routed_model(self, route: str):
        """Placeholder for executing models based on routing decisions."""
        if route == 'EarningsModelRun':
            logging.info("(Placeholder) Would run a discounted cash flow model here.")
        elif route == 'ComplianceCheck':
            logging.info("(Placeholder) Would run a regulatory impact model here.")
        else:
            logging.info("(Placeholder) Would run a general analysis model here.")

    def _run_evaluator(self):
        """Generates the final investment thesis using the evaluator agent."""
        logging.info("Generating final thesis with Evaluator-Optimizer...")
        self.state["final_thesis"] = self.evaluator.run(self.state, self.symbol)

    def _finalize_analysis(self):
        """Finalizes the analysis by updating memory and printing the result."""
        logging.info("Finalizing analysis...")
        if self.state["final_thesis"]:
            self.memory.update(self.symbol, self.state)
            logging.info(f"Memory updated for {self.symbol}.")

        logging.info("\n--- Completed Analysis ---")
        logging.info("Final Thesis:")
        logging.info(self.state["final_thesis"])
        print("\n--- Completed Analysis ---")
        print("Final Thesis:")
        print(self.state["final_thesis"])

    def run(self):
        """Runs the full analysis pipeline."""
        logging.info(f"--- Starting Analysis for {self.symbol} ---")
        if self._generate_plan():
            self._execute_plan()
            self._finalize_analysis()

def main():
    """Main function to run the investment analysis."""
    # Load environment variables and configure APIs
    load_env()
    setup_logging()
    genai.configure(api_key=os.environ.get('GOOGLE_API_KEY'))

    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description="Run an agentic investment analysis for a given stock symbol.")
    parser.add_argument("symbol", type=str, help="The stock symbol to analyze (e.g., NVDA).")
    args = parser.parse_args()

    # Run the analysis
    orchestrator = InvestmentAnalysisOrchestrator(symbol=args.symbol)
    orchestrator.run()

if __name__ == '__main__':
    main()
