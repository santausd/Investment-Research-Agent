import logging
import yfinance as yf
from datetime import datetime, timedelta
import os
from newsapi import NewsApiClient
from fredapi import Fred
from sec_api import QueryApi
import requests
import diskcache
from pathlib import Path
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from utils.docstore import DocumentStore

# Setup a cache directory in the project root
cache_dir = Path(__file__).parent.parent / 'cache'
cache = diskcache.Cache(str(cache_dir))

class ToolboxAgent:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings()
        self.docstore = DocumentStore()

    @cache.memoize(expire=86400, ignore={0}) # Cache for 24 hours, ignore self
    def get_yahoo_finance_data(self, symbol: str) -> dict:
        """Fetches price, P/E, and fundamental metrics from Yahoo Finance."""
        logging.info(f"Fetching data for {symbol} from yfinance (cache miss or expired)")
        try:
            ticker = yf.Ticker(symbol)
            return ticker.info
        except Exception as e:
            logging.error(f"An error occurred with yfinance for symbol {symbol}: {e}")
            return None

    @cache.memoize(expire=86400, ignore={0}) # Cache for 24 hours, ignore self
    def get_financial_news(self, symbol: str) -> dict:
        """Fetches financial news for a given symbol."""
        logging.info(f"Fetching news for {symbol} from newsapi (cache miss or expired)")
        try:
            newsapi = NewsApiClient(api_key=os.environ.get('NEWS_API_KEY'))
            all_articles = newsapi.get_everything(q=symbol,
                                                      language='en',
                                                      sort_by='relevancy',
                                                      page_size=5)
            return all_articles
        except Exception as e:
            logging.error(f"An error occurred with NewsAPI for symbol {symbol}: {e}")
            return None

    @cache.memoize(expire=86400, ignore={0}) # Cache for 24 hours, ignore self
    def get_economic_data(self, indicator: str) -> dict:
        """Fetches economic data from FRED."""
        logging.info(f"Fetching data for {indicator} from FRED (cache miss or expired)")
        try:
            fred = Fred(api_key=os.environ.get('FRED_API_KEY'))
            data = fred.get_series(indicator)
            return data.to_dict()
        except Exception as e:
            logging.error(f"An error occurred with FRED for indicator {indicator}: {e}")
            return None
        
    @cache.memoize(expire=86400, ignore={0}) # Cache for 24 hours, ignore self
    def get_filing_data(self, symbol: str) -> dict:
        """Fetches Filings data from Sec Edgar, chunks it, and stores it in a FAISS vector store."""
        logging.info(f"Fetching filing data for {symbol} from Sec Edgar (cache miss or expired)")
        query = {
            "query": { "query_string": {
                "query": f"ticker:{symbol} AND (formType:\"10-K\" OR formType:\"10-Q\" OR formType:\"8-K\")"
            }},
            "from": "0",
            "size": "5",
            "sort": [{ "filedAt": { "order": "desc" }}]
        }

        try:
            sec = QueryApi(api_key=os.environ.get('SEC_API_KEY'))
            filings = sec.get_filings(query)["filings"]
            
            for filing in filings:
                filing_key = f'{filing["formType"]}_{filing["accessionNo"]}'
                doc_url = filing['linkToFilingDetails']
                
                logging.info(f"  - Found filing: {filing_key}, filed on {filing['filedAt']}")
                
                # Download the filing content
                response = requests.get(doc_url)
                response.raise_for_status()
                
                # Parse and split the document into meaningful chunks
                # This is a simplified example. A more robust solution would use a proper HTML parser
                # and identify sections based on headings.
                text = response.text
                sections = text.split("\n\n") # Simple split by double newline
                
                # Create documents for each section
                docs = [Document(page_content=section) for section in sections if section.strip()]
                
                if not docs:
                    logging.warning(f"    - No content found for filing {filing_key}")
                    continue

                # Create and save the FAISS index
                vectorstore = FAISS.from_documents(docs, self.embeddings)
                index_path = Path(__file__).parent.parent / f"faiss_indexes/{symbol}/{filing_key}"
                index_path.parent.mkdir(parents=True, exist_ok=True)
                vectorstore.save_local(str(index_path))
                logging.info(f"    - Saved FAISS index to {index_path}")

            return {"status": "success", "message": "Filings downloaded and indexed."}
        except Exception as e:
            logging.error(f"An error occurred with Sec Edgar for symbol {symbol}: {e}")
            return None

    def query_sec_filings(self, symbol: str, query: str) -> list:
        """Queries the SEC filings for a given symbol."""
        logging.info(f"Querying SEC filings for {symbol} with query: {query}")
        try:
            symbol_index_path = Path(__file__).parent.parent / f"faiss_indexes/{symbol}"
            if not symbol_index_path.exists():
                return ["No SEC filings found for this symbol."]

            # Load all FAISS indexes for the symbol
            vectorstores = []
            for index_path in symbol_index_path.iterdir():
                if index_path.is_dir():
                    try:
                        vectorstores.append(FAISS.load_local(str(index_path), self.embeddings, allow_dangerous_deserialization=True))
                    except Exception as e:
                        logging.error(f"Error loading index {index_path}: {e}")

            if not vectorstores:
                return ["No valid SEC filing indexes found for this symbol."]

            # Perform similarity search across all loaded vectorstores
            all_docs = []
            for vectorstore in vectorstores:
                all_docs.extend(vectorstore.similarity_search(query))

            # Sort the results by relevance (score) if available, otherwise just return the content
            # Note: FAISS similarity_search doesn't directly return scores in this form.
            # For more advanced ranking, you might need to use `similarity_search_with_score`.
            return [doc.page_content for doc in all_docs]
        except Exception as e:
            logging.error(f"An error occurred while querying SEC filings for symbol {symbol}: {e}")
            return []

    def fetch(self, tool_name: str, **kwargs) -> dict:
        """Dynamically dispatches to the correct tool wrapper."""
        logging.info(f"Dispatching to tool: {tool_name} with params: {kwargs}")
        if tool_name == 'yfinance':
            return self.get_yahoo_finance_data(symbol=kwargs.get('symbol'))
        elif tool_name == 'newsapi':
            return self.get_financial_news(symbol=kwargs.get('symbol'))
        elif tool_name == 'fred':
            return self.get_economic_data(indicator=kwargs.get('indicator'))
        elif tool_name == 'secEdgar':
            return self.get_filing_data(symbol=kwargs.get('symbol'))
        elif tool_name == 'query_sec_filings':
            return self.query_sec_filings(symbol=kwargs.get('symbol'), query=kwargs.get('query'))
        else:
            logging.warning(f"Tool {tool_name} not recognized.")
            return None
