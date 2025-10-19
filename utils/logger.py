from datetime import datetime

class AgentLogger:
    def __init__(self, state):
        self.state = state
        self.state.setdefault("conversation_logs", [])

    def log(self, sender, receiver, content, **metadata):
        self.state["conversation_logs"].append({
            "timestamp": datetime.utcnow().isoformat(),
            "sender": sender,
            "receiver": receiver,
            "content": content,
            "metadata": metadata or {}
        })

