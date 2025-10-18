import yfinance as yf
from datetime import datetime, timedelta
import os
from newsapi import NewsApiClient
from fredapi import Fred
from sec_api import QueryApi
import requests
import diskcache
from pathlib import Path
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.logger_config import logger

# Setup a cache directory in the project root
cache_dir = Path(__file__).parent.parent / 'cache'
cache = diskcache.Cache(str(cache_dir))

class ToolboxAgent:
    def __init__(self):
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=128)
        self.indices = {}
        self.documents = {}

    @cache.memoize(expire=86400, ignore={0}) # Cache for 24 hours, ignore self
    def get_yahoo_finance_data(self, symbol: str) -> dict:
        """Fetches price, P/E, and fundamental metrics from Yahoo Finance."""
        logger.info(f"Fetching data for {symbol} from yfinance (cache miss or expired)")
        try:
            ticker = yf.Ticker(symbol)
            return ticker.info
        except Exception as e:
            logger.error(f"An error occurred with yfinance for symbol {symbol}: {e}")
            return None

    @cache.memoize(expire=86400, ignore={0}) # Cache for 24 hours, ignore self
    def get_financial_news(self, symbol: str) -> dict:
        """Fetches financial news for a given symbol."""
        logger.info(f"Fetching news for {symbol} from newsapi (cache miss or expired)")
        try:
            newsapi = NewsApiClient(api_key=os.environ.get('NEWS_API_KEY'))
            all_articles = newsapi.get_everything(q=symbol,
                                                      language='en',
                                                      sort_by='relevancy',
                                                      page_size=5)
            return all_articles
        except Exception as e:
            logger.error(f"An error occurred with NewsAPI for symbol {symbol}: {e}")
            return None

    @cache.memoize(expire=86400, ignore={0}) # Cache for 24 hours, ignore self
    def get_economic_data(self, indicator: str) -> dict:
        """Fetches economic data from FRED."""
        logger.info(f"Fetching data for {indicator} from FRED (cache miss or expired)")
        try:
            fred = Fred(api_key=os.environ.get('FRED_API_KEY'))
            data = fred.get_series(indicator)
            return data.to_dict()
        except Exception as e:
            logger.error(f"An error occurred with FRED for indicator {indicator}: {e}")
            return None
        
    @cache.memoize(expire=86400, ignore={0}) # Cache for 24 hours, ignore self
    def get_filing_data(self, symbol: str) -> dict:
        """Fetches filings from SEC Edgar, processes them, and stores them in a FAISS index."""
        logger.info(f"Fetching and processing filing data for {symbol} from Sec Edgar.")
        query = {
            "query": { "query_string": {
                "query": f"ticker:{symbol} AND (formType:\"10-K\" OR formType:\"10-Q\")",
                "time_zone": "America/New_York"
            }},
            "from": "0",
            "size": "5",
            "sort": [{ "filedAt": { "order": "desc" }}]
        }

        try:
            sec = QueryApi(api_key=os.environ.get('SEC_API_KEY'))
            filings = sec.get_filings(query)["filings"]
            if not filings:
                logger.warning("No filings found.")
                return {"status": "No filings found."}

            all_chunks = []
            for filing in filings:
                doc_url = filing['linkToTxt']
                logger.info(f"  - Processing filing: {filing['formType']} from {doc_url}")
                
                try:
                    response = requests.get(doc_url)
                    response.raise_for_status()
                    raw_text = response.text
                except requests.exceptions.RequestException as e:
                    logger.error(f"    - Could not download filing content: {e}")
                    continue

                chunks = self.text_splitter.split_text(raw_text)
                all_chunks.extend(chunks)

            if not all_chunks:
                logger.warning("No content to process.")
                return {"status": "No content to process."}

            embeddings = self.embedding_model.encode(all_chunks)
            embedding_dim = embeddings.shape[1]
            index = faiss.IndexFlatL2(embedding_dim)
            index.add(np.array(embeddings, dtype=np.float32))
            
            self.indices[symbol] = index
            self.documents[symbol] = all_chunks

            return {"status": f"Successfully processed and stored {len(filings)} filings."}
        except Exception as e:
            logger.error(f"An error occurred with Sec Edgar processing for symbol {symbol}: {e}")
            return None

    def query_sec_filings(self, query: str, symbol: str) -> list:
        """Queries the vectorized SEC filings for a given symbol."""
        if symbol not in self.indices:
            logger.warning("No SEC filings have been processed for this symbol yet.")
            return ["No SEC filings have been processed for this symbol yet."]

        index = self.indices[symbol]
        documents = self.documents[symbol]
        
        query_embedding = self.embedding_model.encode([query])
        distances, indices = index.search(np.array(query_embedding, dtype=np.float32), k=5)
        
        results = [documents[i] for i in indices[0]]
        return results

    def fetch(self, tool_name: str, **kwargs) -> dict:
        """Dynamically dispatches to the correct tool wrapper."""
        logger.info(f"Dispatching to tool: {tool_name} with params: {kwargs}")
        if tool_name == 'yfinance':
            return self.get_yahoo_finance_data(symbol=kwargs.get('symbol'))
        elif tool_name == 'newsapi':
            return self.get_financial_news(symbol=kwargs.get('symbol'))
        elif tool_name == 'fred':
            return self.get_economic_data(indicator=kwargs.get('indicator'))
        elif tool_name == 'secEdgar':
            return self.get_filing_data(symbol=kwargs.get('symbol'))
        elif tool_name == 'query_sec_filings':
            return self.query_sec_filings(query=kwargs.get('query'), symbol=kwargs.get('symbol'))
        else:
            logger.warning(f"Tool {tool_name} not recognized.")
            return None

