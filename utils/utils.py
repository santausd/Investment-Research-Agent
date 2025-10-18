import os
from utils.logger_config import logger

def load_env(filepath="config/aai_520_proj.config"):
    """
    Loads environment variables from the aai_520_project.config.
    Each line in the file should be in the format KEY=VALUE.
    """
    try:
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    key, value = line.split('=', 1)
                    os.environ[key] = value
        logger.info(f"Environment variables loaded from {filepath}")
    except FileNotFoundError:
        logger.error(f"Error: Config file not found at {filepath}. Make sure it is in the project config directory.")
    except Exception as e:
        logger.error(f"Error loading environment variables from {filepath}: {e}")
