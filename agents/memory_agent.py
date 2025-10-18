import json
from datetime import datetime
from utils.logger_config import logger

class MemoryAgent:
    def __init__(self, db_path='memory_db.json'):
        self.db_path = db_path
        self.memory = self._load_memory()

    def _load_memory(self):
        try:
            with open(self.db_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}

    def _save_memory(self):
        with open(self.db_path, 'w') as f:
            json.dump(self.memory, f, indent=4)

    def retrieve(self, symbol: str) -> dict:
        """Retrieves the memory for a given symbol."""
        return self.memory.get(symbol)

    def update(self, symbol: str, analysis_state: dict):
        """Updates the memory for a given symbol with the entire analysis state."""
        self.memory[symbol] = {
            'state': analysis_state,
            'date': datetime.now().isoformat()
        }
        self._save_memory()
        logger.info(f"Memory updated for {symbol}")
