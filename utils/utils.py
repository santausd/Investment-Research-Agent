import os
from dotenv import load_dotenv
import logging.config

def load_env():
    """Loads environment variables from .env file."""
    load_dotenv(dotenv_path='config/aai_520_proj.config')

def setup_logging():
    """Loads logging configuration from file."""
    logging.config.fileConfig('config/logging.config')
