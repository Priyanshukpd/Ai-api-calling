import json
from typing import Dict, List, Optional
from utils.logger import setup_logger

logger = setup_logger("Registry")

class ToolRegistry:
    def __init__(self, config_path: str, llm=None):
        self.tools: Dict[str, dict] = {}
        self.embeddings: Dict[str, List[float]] = {}
        self.llm = llm # LLMService instance needed for embeddings
        self.load_tools(config_path)
        
        if self.llm:
            self._index_tools()

    def load_tools(self, path: str):
        try:
            with open(path, 'r') as f:
                data = json.load(f)
                for tool in data:
                    self.tools[tool['name']] = tool
            logger.info(f"Loaded {len(self.tools)} tools from {path}")
        except Exception as e:
            logger.error(f"Failed to load tools: {e}")

    def _index_tools(self):
        """Generates embeddings for all loaded tools."""
        logger.info("Indexing tools (generating embeddings)...")
        for name, tool in self.tools.items():
            # Create a rich description for embedding
            text = f"{name}: {tool['description']} Parameters: {', '.join(tool['parameters'].keys())}"
            vector = self.llm.get_embedding(text)
            if vector:
                self.embeddings[name] = vector
        logger.info(f"Indexed {len(self.embeddings)} tools.")

    def get_tool(self, name: str) -> Optional[dict]:
        return self.tools.get(name)

    def search_tools(self, query: str) -> List[dict]:
        """Hybrid Search: Tries Semantic first, falls back to Keyword."""
        if self.llm and self.embeddings:
            return self._search_semantic(query)
        return self._search_keyword(query)

    def _search_semantic(self, query: str) -> List[dict]:
        query_vector = self.llm.get_embedding(query)
        if not query_vector:
            return self._search_keyword(query)

        scores = []
        for name, tool_vector in self.embeddings.items():
            score = self._cosine_similarity(query_vector, tool_vector)
            scores.append((score, self.tools[name]))
        
        # Sort by score (Highest first)
        scores.sort(key=lambda x: x[0], reverse=True)
        
        # Return top 3
        top_tools = [s[1] for s in scores[:3]]
        logger.info(f"Semantic Search for '{query}' found: {[t['name'] for t in top_tools]}")
        return top_tools

    def _cosine_similarity(self, v1: List[float], v2: List[float]) -> float:
        dot_product = sum(a * b for a, b in zip(v1, v2))
        norm_v1 = sum(a * a for a in v1) ** 0.5
        norm_v2 = sum(a * a for a in v2) ** 0.5
        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0
        return dot_product / (norm_v1 * norm_v2)

    def _search_keyword(self, query: str) -> List[dict]:
        # Fallback to old keyword logic
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
        return [r[1] for r in results[:3]]
