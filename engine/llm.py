from abc import ABC, abstractmethod
from typing import List, Dict, Any
import json
import re

class LLMService(ABC):
    @abstractmethod
    def predict(self, system_prompt: str, user_prompt: str) -> str:
        pass



class OllamaLLM(LLMService):
    def __init__(self, model: str = "gemma3:270m", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url
        import requests 
        self.requests = requests

    def predict(self, system_prompt: str, user_prompt: str) -> str:
        url = f"{self.base_url}/api/generate"
        
        # Construct a prompt that encourages JSON
        full_prompt = f"{system_prompt}\n\nUser Input: {user_prompt}\n\nResponse (JSON only):"
        
        payload = {
            "model": self.model,
            "prompt": full_prompt,
            "stream": False,
            "format": "json" # Force JSON mode
        }
        
        try:
            response = self.requests.post(url, json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()
            return result.get("response", "")
        except Exception as e:
            return f"Error calling Ollama: {str(e)}"

class TogetherLLM(LLMService):
    def __init__(self, api_key: str, model: str = "ServiceNow-AI/Apriel-1.6-15b-Thinker"):
        self.api_key = api_key
        self.model = model
        from together import Together
        self.client = Together(api_key=api_key)

    def predict(self, messages: List[Dict[str, str]]) -> str:
        try:
            # Ensure the model supports chat format (Apriel does)
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=False
            )
            content = response.choices[0].message.content
            return self._clean_response(content)
        except Exception as e:
            return f"Error calling Together AI: {str(e)}"

    def _clean_response(self, content: str) -> str:
        # Remove thinking tags if present
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
        
        # Attempt to find JSON block
        match = re.search(r'\{.*\}', content, re.DOTALL)
        if match:
            return match.group(0)
        return content.strip()
