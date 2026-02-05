import sys
import os
# Add current directory to path so imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from engine.registry import ToolRegistry
from engine.llm import MockLLM, OllamaLLM
from engine.orchestrator import Orchestrator
from utils.logger import setup_logger

logger = setup_logger("Main")

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, 'config', 'api_definitions.json')
    
from engine.registry import ToolRegistry
from engine.llm import OllamaLLM, TogetherLLM
from engine.orchestrator import Orchestrator
from utils.logger import setup_logger

logger = setup_logger("Main")

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, 'config', 'api_definitions.json')
    
    registry = ToolRegistry(config_path)
    
    # API Key provided by user
    api_key = "bff39f38ee07df9a08ff8d2e7279b9d7223ab3f283a30bc39590d36f77dbd2fd"
    llm = TogetherLLM(api_key=api_key)
    
    orchestrator = Orchestrator(registry, llm)
    
    print("Agent Initialized. Type 'quit' to exit.")
    print("Try saying: 'I want to update my nominee'")
    
    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in ['quit', 'exit']:
                break
            
            response = orchestrator.process_message(user_input)
            
            # Check for visible reasoning
            if "[THOUGHT]" in response and "[RESPONSE]" in response:
                parts = response.split("[RESPONSE]")
                thought_part = parts[0].replace("[THOUGHT]", "").strip()
                final_response = parts[1].strip()
                
                # Print Thought in a distinct color/format
                print(f"\n\033[90mAgent Thought: {thought_part}\033[0m") # Grey color
                print(f"Agent: {final_response}")
            else:
                print(f"Agent: {response}")
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    main()
