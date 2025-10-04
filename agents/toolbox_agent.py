import yfinance as yf
from datetime import datetime, timedelta
import os
from newsapi import NewsApiClient
from fredapi import Fred



class ToolboxAgent:
    def __init__(self):
        self.cache = {}
        self.newsapi = NewsApiClient(api_key=os.environ.get('NEWS_API_KEY'))
        self.fred = Fred(api_key=os.environ.get('FRED_API_KEY'))

    def _is_cache_valid(self, symbol, tool_name):
        if symbol in self.cache and tool_name in self.cache[symbol]:
            timestamp = self.cache[symbol][tool_name]['timestamp']
            if datetime.now() - timestamp < timedelta(hours=24):
                return True
        return False

    def get_yahoo_finance_data(self, symbol: str) -> dict:
        """Fetches price, P/E, and fundamental metrics from Yahoo Finance."""
        tool_name = 'yfinance'
        if self._is_cache_valid(symbol, tool_name):
            print(f"Returning cached data for {symbol} from {tool_name}")
            return self.cache[symbol][tool_name]['data']

        try:
            print(f"Fetching data for {symbol} from {tool_name}")
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            if symbol not in self.cache:
                self.cache[symbol] = {}
            self.cache[symbol][tool_name] = {
                'timestamp': datetime.now(),
                'data': info
            }
            
            return info
        except Exception as e:
            print(f"An error occurred with yfinance for symbol {symbol}: {e}")
            return None

    def get_financial_news(self, symbol: str) -> dict:
        """Fetches financial news for a given symbol."""
        tool_name = 'newsapi'
        if self._is_cache_valid(symbol, tool_name):
            print(f"Returning cached news for {symbol}")
            return self.cache[symbol][tool_name]['data']

        try:
            print(f"Fetching news for {symbol}")
            all_articles = self.newsapi.get_everything(q=symbol,
                                                      language='en',
                                                      sort_by='relevancy',
                                                      page_size=5)
            if symbol not in self.cache:
                self.cache[symbol] = {}
            self.cache[symbol][tool_name] = {
                'timestamp': datetime.now(),
                'data': all_articles
            }
            return all_articles
        except Exception as e:
            print(f"An error occurred with NewsAPI for symbol {symbol}: {e}")
            return None

    def get_economic_data(self, indicator: str) -> dict:
        """Fetches economic data from FRED."""
        tool_name = 'fred'
        if self._is_cache_valid(indicator, tool_name):
            print(f"Returning cached data for {indicator} from {tool_name}")
            return self.cache[indicator][tool_name]['data']

        try:
            print(f"Fetching data for {indicator} from {tool_name}")
            data = self.fred.get_series(indicator)
            
            if indicator not in self.cache:
                self.cache[indicator] = {}
            self.cache[indicator][tool_name] = {
                'timestamp': datetime.now(),
                'data': data.to_dict()
            }
            return data.to_dict()
        except Exception as e:
            print(f"An error occurred with FRED for indicator {indicator}: {e}")
            return None

    def fetch(self, tool_name: str, symbol: str) -> dict:
        """Dynamically dispatches to the correct tool wrapper."""
        if tool_name == 'yfinance':
            return self.get_yahoo_finance_data(symbol)
        elif tool_name == 'newsapi':
            return self.get_financial_news(symbol)
        elif tool_name == 'fred':
            return self.get_economic_data(symbol)
        else:
            print(f"Tool {tool_name} not recognized.")
            return None
