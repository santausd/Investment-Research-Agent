import yfinance as yf
from datetime import datetime, timedelta
import os
from newsapi import NewsApiClient
from fredapi import Fred
from sec_api import QueryApi
import requests
import traceback

from utils.logger import AgentLogger

class ToolboxAgent:
    def __init__(self):
        self.cache = {}
        self.newsapi = NewsApiClient(api_key=os.environ.get('NEWS_API_KEY'))
        self.fred = Fred(api_key=os.environ.get('FRED_API_KEY'))
        self.sec = QueryApi(api_key=os.environ.get('SEC_API_KEY'))

    def _is_cache_valid(self, symbol, tool_name):
        if symbol in self.cache and tool_name in self.cache[symbol]:
            timestamp = self.cache[symbol][tool_name]['timestamp']
            if datetime.now() - timestamp < timedelta(hours=24):
                return True
        return False

    # Helper to initialize logger only once per symbol/session
    def _get_logger(self, state):
        return AgentLogger(state)

    # -----------------------------------------------------------------------------------
    # YFinance Data
    # -----------------------------------------------------------------------------------
    def get_yahoo_finance_data(self, symbol: str, state: dict) -> dict:
        """Fetches price, P/E, and fundamental metrics from Yahoo Finance."""
        tool_name = 'yfinance'
        logger = self._get_logger(state)

        if self._is_cache_valid(symbol, tool_name):
            print(f"Returning cached data for {symbol} from {tool_name}")
            logger.log("ToolboxAgent", tool_name, f"Cache hit for {symbol}")
            return self.cache[symbol][tool_name]['data']

        try:
            print(f"Fetching data for {symbol} from {tool_name}")
            logger.log("ToolboxAgent", tool_name, f"Fetching Yahoo Finance data for {symbol}")
            ticker = yf.Ticker(symbol)
            info = ticker.info

            if symbol not in self.cache:
                self.cache[symbol] = {}
            self.cache[symbol][tool_name] = {
                'timestamp': datetime.now(),
                'data': info
            }

            logger.log(tool_name, "ToolboxAgent", f"Successfully fetched data for {symbol}")
            return info
        except Exception as e:
            error_details = traceback.format_exc()
            logger.log("ToolboxAgent", tool_name, f"Error fetching yfinance data for {symbol}: {e}", level="error", traceback=error_details)
            print(f" YFinance Error for {symbol}: {e}")
            return None

    # -----------------------------------------------------------------------------------
    # Financial News
    # -----------------------------------------------------------------------------------
    def get_financial_news(self, symbol: str, state: dict) -> dict:
        """Fetches financial news for a given symbol."""
        tool_name = 'newsapi'
        logger = self._get_logger(state)

        if self._is_cache_valid(symbol, tool_name):
            print(f"Returning cached news for {symbol}")
            logger.log("ToolboxAgent", tool_name, f"Cache hit for {symbol} (news)")
            return self.cache[symbol][tool_name]['data']

        try:
            print(f"Fetching news for {symbol}")
            logger.log("ToolboxAgent", tool_name, f"Fetching news for {symbol}")
            all_articles = self.newsapi.get_everything(
                q=symbol,
                language='en',
                sort_by='relevancy',
                page_size=5
            )
            if symbol not in self.cache:
                self.cache[symbol] = {}
            self.cache[symbol][tool_name] = {
                'timestamp': datetime.now(),
                'data': all_articles
            }

            logger.log(tool_name, "ToolboxAgent", f"Fetched {len(all_articles.get('articles', []))} news articles for {symbol}")
            return all_articles

        except Exception as e:
            error_details = traceback.format_exc()
            print(f" NewsAPI Error for {symbol}: {e}")
            logger.log("ToolboxAgent", tool_name, f"Error fetching news for {symbol}: {e}", level="error", traceback=error_details)
            return None

    # -----------------------------------------------------------------------------------
    # Economic Data
    # -----------------------------------------------------------------------------------
    def get_economic_data(self, indicator: str, state: dict) -> dict:
        """Fetches economic data from FRED."""
        tool_name = 'fred'
        logger = self._get_logger(state)
        if self._is_cache_valid(indicator, tool_name):
            print(f"Returning cached data for {indicator} from {tool_name}")
            return self.cache[indicator][tool_name]['data']

        try:
            print(f"Fetching data for {indicator} from {tool_name}")
            logger.log("ToolboxAgent", tool_name, f"Fetching economic data for indicator '{indicator}'")
            data = self.fred.get_series(indicator)

            logger.log(tool_name, "ToolboxAgent", f"Successfully fetched {len(data)} records for {indicator}")

            if indicator not in self.cache:
                self.cache[indicator] = {}
            self.cache[indicator][tool_name] = {
                'timestamp': datetime.now(),
                'data': data.to_dict()
            }
            return data.to_dict()
        except Exception as e:
            error_details = traceback.format_exc()
            print(f"An error occurred with FRED for indicator {indicator}: {e}")
            logger.log("ToolboxAgent", tool_name, f"Error fetching FRED data for {indicator}: {e}", level="error", traceback=error_details)
            return None

    # -----------------------------------------------------------------------------------
    # Filing Data (SEC EDGAR)
    # -----------------------------------------------------------------------------------
    def get_filing_data(self, indicator: str, state: dict) -> dict:
        """Fetches Filings data from Sec Edgar."""
        tool_name = 'secEdgar'
        logger = self._get_logger(state)

        query = {
            "query": (
                f'(formType:"10-K" OR formType:"10-Q" OR formType:"8-K" OR '
                f'formType:"SC 13D" OR formType:"SC 13G") AND ticker:{indicator}'
            ),
            "from": 0,
            "size": 4,
            "sort": [{"filedAt": {"order": "desc"}}]
        }
        print(query)

        logger.log("ToolboxAgent", tool_name, f"Preparing SEC EDGAR query for {indicator}", query=query)

        # Check cache first
        if self._is_cache_valid(indicator, tool_name):
            print(f"Returning cached data for {indicator} from {tool_name}")
            logger.log("ToolboxAgent", tool_name, f"Cache hit for SEC EDGAR filings of {indicator}")
            return self.cache[indicator][tool_name]['data']

        try:
            print(f"Fetching data for {indicator} from {tool_name}")
            logger.log("ToolboxAgent", tool_name, f"Fetching latest SEC filings for {indicator}")
            data = self.sec.get_filings(query)["filings"]

            filingDataRaw = {}
            folder_path = os.path.join("..", "utils", "filingDocuments", indicator)
            os.makedirs(folder_path, exist_ok=True)

            for filing in data:
                folder_path = "..\\utils\\filingDocuments\\"+(indicator)
                os.makedirs(folder_path, exist_ok=True)
                form_type = filing["formType"].replace("/", "-")
                description = filing["description"].replace("/", "-")
                
                fileType = {}

                for doc in filing.get("documentFormatFiles", []):
                    doc_url = doc.get("documentUrl", "")
                    if not doc_url:
                        continue

                    file_ext = os.path.splitext(doc_url)[1]
                    file_name = f"{form_type}-{description}{file_ext}"
                    file_path = os.path.join(folder_path, file_name)

                    if file_ext in [".txt", ".htm", ".html"]:
                        try:
                            response = requests.get(doc_url, timeout=10)
                            response.raise_for_status()
                            with open(file_path, "wb") as f:
                                f.write(response.content)
                            filingDataRaw[file_name] = response.content.decode("utf-8", errors="ignore")

                            logger.log(tool_name, "ToolboxAgent", f"Saved filing {file_name} for {indicator}")
                        except Exception as e:
                            error_details = traceback.format_exc()
                            logger.log("ToolboxAgent", tool_name,
                                       f"Error downloading {file_name} for {indicator}: {e}",
                                       level="error", traceback=error_details)
                    else:
                        logger.log("ToolboxAgent", tool_name, f"Skipping unsupported file type: {file_ext}")

            
            # Update cache after successful fetch
            if indicator not in self.cache:
                self.cache[indicator] = {}
            self.cache[indicator][tool_name] = {
                'timestamp': datetime.now(),
                'data': filingDataRaw
            }
            logger.log(tool_name, "ToolboxAgent", f"Fetched and cached {len(filingDataRaw)} filings for {indicator}")
            return filingDataRaw
        except Exception as e:
            error_details = traceback.format_exc()
            print(f" SEC EDGAR Error for {indicator}: {e}")
            logger.log("ToolboxAgent", tool_name,
                       f"Error fetching filings for {indicator}: {e}",
                       level="error", traceback=error_details)
            return None

    def fetch(self, tool_name: str, symbol: str, state: dict) -> dict:
        """Dynamically dispatches to the correct tool wrapper."""
        if tool_name == 'yfinance':
            return self.get_yahoo_finance_data(symbol, state)
        elif tool_name == 'newsapi':
            return self.get_financial_news(symbol, state)
        elif tool_name == 'fred':
            return self.get_economic_data(symbol, state)
        elif tool_name == 'secEdgar':
            return self.get_filing_data(symbol, state)
        else:
            print(f"Tool {tool_name} not recognized.")
            logger.log(tool_name, "ToolboxAgent", f"Tool {tool_name} not recognized")
            return None
