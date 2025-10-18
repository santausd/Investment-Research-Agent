import json
import os
import traceback
from datetime import datetime
from utils.logger import AgentLogger


class MemoryAgent:
    def __init__(self, db_path='memory_db.json'):
        self.db_path = db_path
        self.memory = self._load_memory()

    # ------------------------------------------------------------
    # Internal helper to attach logger
    # ------------------------------------------------------------
    def _get_logger(self, state):
        return AgentLogger(state) if state and "conversation_logs" in state else None

    # ------------------------------------------------------------
    # Load memory from JSON file
    # ------------------------------------------------------------
    def _load_memory(self):
        try:
            if not os.path.exists(self.db_path):
                return {}
            with open(self.db_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f" Failed to load memory DB: {e}")
            return {}

    # ------------------------------------------------------------
    # Save memory to disk
    # ------------------------------------------------------------
    def _save_memory(self):
        try:
            with open(self.db_path, 'w') as f:
                json.dump(self.memory, f, indent=4)
        except Exception as e:
            print(f" Failed to save memory DB: {e}")

    # ------------------------------------------------------------
    # Retrieve stored memory
    # ------------------------------------------------------------
    def retrieve(self, symbol: str, state: dict = None) -> dict:
        """Retrieves memory for a given stock symbol."""
        logger = self._get_logger(state)
        try:
            memory_entry = self.memory.get(symbol)
            if memory_entry:
                if logger:
                    logger.log("MemoryAgent", "System", f"Retrieved memory for {symbol}")
                return memory_entry
            else:
                if logger:
                    logger.log("MemoryAgent", "System", f"No memory found for {symbol}")
                return None
        except Exception as e:
            error_details = traceback.format_exc()
            if logger:
                logger.log("MemoryAgent", "System",
                           f"Error retrieving memory for {symbol}: {e}",
                           level="error", traceback=error_details)
            return None

    # ------------------------------------------------------------
    # Update memory
    # ------------------------------------------------------------
    def update(self, symbol: str, final_analysis: dict, state: dict = None):
        """Updates or creates a memory entry for a stock symbol."""
        logger = self._get_logger(state)

        try:
            if symbol not in self.memory:
                self.memory[symbol] = {}

            self.memory[symbol] = {
                'summary': final_analysis.get('summary', ''),
                'key_metrics': final_analysis.get('key_metrics', {}),
                'date': datetime.now().isoformat()
            }

            self._save_memory()

            msg = f"Memory updated for {symbol}"
            print(msg)
            if logger:
                logger.log("MemoryAgent", "System", msg, payload=self.memory[symbol])

        except Exception as e:
            error_details = traceback.format_exc()
            print(f" Error updating memory for {symbol}: {e}")
            if logger:
                logger.log("MemoryAgent", "System",
                           f"Error updating memory for {symbol}: {e}",
                           level="error", traceback=error_details)

