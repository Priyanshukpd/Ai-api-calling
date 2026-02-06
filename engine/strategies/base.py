from abc import ABC, abstractmethod
from typing import Any, Optional

class BaseStrategy(ABC):
    """Base class for all processing strategies."""
    
    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        self.registry = orchestrator.registry
        self.llm = orchestrator.llm
        self.settings = orchestrator.settings
    
    @abstractmethod
    def can_handle(self, session: dict, message: str) -> bool:
        """Return True if this strategy can handle the request."""
        pass
    
    @abstractmethod
    def process(self, session: dict, message: str) -> Optional[str]:
        """Process the message and return a response, or None to pass to next strategy."""
        pass
