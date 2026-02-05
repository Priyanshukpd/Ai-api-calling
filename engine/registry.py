import json
from typing import Dict, List, Optional
from utils.logger import setup_logger

logger = setup_logger("Registry")

class ToolRegistry:
    def __init__(self, config_path: str):
        self.tools: Dict[str, dict] = {}
        self.load_tools(config_path)

    def load_tools(self, path: str):
        try:
            with open(path, 'r') as f:
                data = json.load(f)
                for tool in data:
                    self.tools[tool['name']] = tool
            logger.info(f"Loaded {len(self.tools)} tools from {path}")
        except Exception as e:
            logger.error(f"Failed to load tools: {e}")

    def get_tool(self, name: str) -> Optional[dict]:
        return self.tools.get(name)

    def search_tools(self, query: str) -> List[dict]:
        # Simple keyword match for now. In production, use embeddings/RAG.
        # This is where the 'Universal' aspect shines - scaling this search.
        results = []
        query_terms = query.lower().split()
        for name, tool in self.tools.items():
            score = 0
            text = (name + " " + tool['description']).lower()
            for term in query_terms:
                if term in text:
                    score += 1
            if score > 0:
                results.append((score, tool))
        
        results.sort(key=lambda x: x[0], reverse=True)
        return [r[1] for r in results[:3]] # Return top 3
