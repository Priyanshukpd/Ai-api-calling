import sys
import os
import time
import json

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from engine.registry import ToolRegistry
from engine.llm import TogetherLLM
from engine.orchestrator import Orchestrator

def test_chaining():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, 'config', 'api_definitions.json')
    registry = ToolRegistry(config_path)
    api_key = "bff39f38ee07df9a08ff8d2e7279b9d7223ab3f283a30bc39590d36f77dbd2fd"
    llm = TogetherLLM(api_key=api_key)
    orchestrator = Orchestrator(registry, llm)

    print("\n--- Testing Autonomous Chaining ---")
    print("Goal: Retrieve Policy Number -> Use it to Update Nominee (2 Steps)")
    
    # Complex multi-step request
    msg1 = "My name is John Doe. Find my policy details."
    print(f"\nYou: {msg1}")
    resp1 = orchestrator.process_message(msg1)
    print(f"Agent: {resp1}")
    
    if "555-999-000" in resp1:
        print("SUCCESS Step 1: Policy Found.")
    else:
        print("FAILURE Step 1: Policy NOT Found.")
        
    # Step 2: Now ask to update, implying use the found number
    msg2 = "Great, now update the nominee to Alice (Wife)"
    print(f"\nYou: {msg2}")
    resp2 = orchestrator.process_message(msg2)
    print(f"Agent: {resp2}")
    
    # Check if it automatically picked up 555-999-000 from the previous turn's output
    if "555-999-000" in resp2:
        print("SUCCESS Step 2: Agent used the discovered Policy Number!")
    else:
        print("FAILURE Step 2: Agent failed to link the context.")

if __name__ == "__main__":
    test_chaining()
