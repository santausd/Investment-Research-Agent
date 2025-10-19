from utils.logger import AgentLogger
import traceback

class RoutingAgent:
    def __init__(self):
        pass

    def _get_logger(self, state):
        """Attach logger if conversation state is provided."""
        return AgentLogger(state) if state and "conversation_logs" in state else None

    def route(self, classification: str, state: dict = None) -> str:
        """Determines the next agent path based on classification with structured logging."""
        logger = self._get_logger(state)

        try:
            if not classification or not isinstance(classification, str):
                msg = "Invalid or empty classification received."
                if logger:
                    logger.log("RoutingAgent", "System", msg, level="error")
                return "GeneralAnalysis"

            normalized_class = classification.lower().strip()
            if logger:
                logger.log(
                    "RoutingAgent",
                    "System",
                    f"Received classification: '{classification}'",
                    payload={"normalized_class": normalized_class}
                )

            if 'earnings' in normalized_class:
                route = 'EarningsModelRun'
            elif 'regulation' in normalized_class:
                route = 'ComplianceCheck'
            elif 'product launch' in normalized_class or 'launch' in normalized_class:
                route = 'MarketImpactAnalysis'
            else:
                route = 'GeneralAnalysis'

            if logger:
                logger.log(
                    "RoutingAgent",
                    "System",
                    f"Routing decision: {route}",
                    payload={"classification": classification, "next_route": route}
                )

            return route

        except Exception as e:
            error_details = traceback.format_exc()
            if logger:
                logger.log("RoutingAgent", "System",
                           f"Routing error: {e}",
                           level="error",
                           traceback=error_details)
            return "GeneralAnalysis"

