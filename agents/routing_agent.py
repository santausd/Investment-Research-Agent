class RoutingAgent:
    def __init__(self):
        pass

    def route(self, classification: str) -> str:
        """Determines the next step based on the classification."""
        
        classification = classification.lower()
        
        if 'earnings' in classification:
            return 'EarningsModelRun'
        elif 'regulation' in classification:
            return 'ComplianceCheck'
        elif 'product launch' in classification:
            return 'MarketImpactAnalysis'
        else:
            return 'GeneralAnalysis'
