Prerequisites:
1. python 3.12

Steps for running the program:
1. Create a virtual environment.
2. Change folder to Investment-Research-Agent
3. Install relevant packages pip install -r requirement.txt
4. Update all keys in aai_520_proj.config. Subscribe token from newsfeed, fred etc after creating a free account. Use GEMINI_MODEL_NAME=gemini-2.5-flash
Get the "NEWS_API_KEY" API Token from https://newsapi.org/
Get the "FRED_API_KEY" API Token from https://fred.stlouisfed.org/docs/api/api_key.html
Get the "GOOGLE_API_KEY" token from https://console.google.com/. Menu : API & Service --> Credentials. Click on Create Credential button. Select "Application Restriction" to None and API Restriction to "Don't Restrict any API".
5. Run python3 main.py.
