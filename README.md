Investment Research Agent 
=========================

This project is a multi-agent investment research system.

Files:
 - app.py                     Simple GUI using streamlit
 - main.py                    Entry point (runs the workflow)
 - config/aai_520_proj.config Project configuration (API keys, model, etc.)
 - utils/llm_integration.py   LLM configuration
 - utils/utils.py             Load environment variables
 - utils/logger.py            Logging tool to log interaction between agents
 - agents/*.py                Planner, Toolbox(News, Earnings, Market), Prompt chaining, Routing, Evaluator agents
 - evaluation/evaluator.py    Grador agent

Setup:
 1. Create a virtual environment
 2. Install required packages `pip install -r requirements.txt`
 3. Update all keys in aai_520_proj.config. Subscribe token from newsfeed, fred etc after creating a free account. 
 4. Use GEMINI_MODEL_NAME=gemini-2.5-flash
 5. Get the "NEWS_API_KEY" API Token from https://newsapi.org/
 6. Get the "FRED_API_KEY" API Token from https://fred.stlouisfed.org/docs/api/api_key.html
 7. Get the "GOOGLE_API_KEY" token from https://console.google.com/. Menu : API & Service --> Credentials. Click on Create Credential button. Select "Application Restriction" to None and API Restriction to "Don't Restrict any API".
 8. Get the "OPENAI_API_KEY" from https://platform.openai.com/api-keys

Run in command line:
`python3 main.py`

Run in GUI using streamlit:
`streamlit run app.py`
