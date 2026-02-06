import json
from typing import Optional
from engine.strategies.base import BaseStrategy
from utils.logger import setup_logger

logger = setup_logger("FastPath")

class FastPathStrategy(BaseStrategy):
    """
    Fast path for simple, direct requests.
    Skips the full ReAct loop when the request clearly matches ONE tool.
    """
    
    COMPLEXITY_MARKERS = [" and ", " then ", " also ", " after that", "first ", "second "]
    
    def can_handle(self, session: dict, message: str) -> bool:
        """Return True if this is a simple, single-tool request."""
        # Skip if complex request
        is_complex = any(marker in message.lower() for marker in self.COMPLEXITY_MARKERS)
        if is_complex:
            logger.info("FastPath: Complex request detected, skipping")
            return False
        
        # Check if we have a matching tool
        matching_tools = self.registry.search_tools(message)
        return len(matching_tools) > 0
    
    def process(self, session: dict, message: str) -> Optional[str]:
        """Process a simple request directly."""
        matching_tools = self.registry.search_tools(message)
        
        if not matching_tools:
            return None
        
        top_tool = matching_tools[0]
        logger.info(f"FastPath: Using tool {top_tool['name']}")
        
        session["current_tool"] = top_tool
        session["state"] = {p: None for p in top_tool['parameters'].keys()}
        
        # Extract slots
        self.orchestrator._extract_slots(session)
        
        # Check for missing required slots
        missing = [p for p, v in session["state"].items() 
                   if v is None and top_tool['parameters'][p]['required']]
        
        if missing:
            logger.info(f"FastPath: Missing slots: {missing}")
            response = self.orchestrator._generate_response(session, missing)
            session["history"].append({"role": "assistant", "content": response})
            return response
        
        # Check safety
        if self.orchestrator._needs_confirmation(session):
            session["waiting_for_confirmation"] = True
            summary = json.dumps({k: v for k, v in session["state"].items() if v})
            response = f"[FAST] Ready to execute {top_tool['name']} with: {summary}. Confirm? (Yes/No)"
            session["history"].append({"role": "assistant", "content": response})
            return response
        
        # Execute directly
        logger.info(f"FastPath: Executing {top_tool['name']}")
        result = self.orchestrator._execute_tool(session)
        session["history"].append({"role": "assistant", "content": f"[FAST] {result}"})
        return f"[FAST] {result}"
