import json
from typing import Dict, Any, List
from engine.registry import ToolRegistry
from engine.llm import LLMService
from utils.logger import setup_logger

logger = setup_logger("Orchestrator")

class Orchestrator:
    def __init__(self, registry: ToolRegistry, llm: LLMService):
        self.registry = registry
        self.llm = llm
        self.state = {} # Context memory: {"tool_name": {param: value}}
        self.history = [] # Conversational memory: [{"role": "user", "content": "..."}, ...]
        self.current_tool = None
        self.waiting_for_confirmation = False
        
        # Load Settings
        import os
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        settings_path = os.path.join(base_dir, 'config', 'agent_settings.json')
        try:
            with open(settings_path, 'r') as f:
                self.settings = json.load(f)
        except Exception:
            logger.warning("Could not load agent_settings.json, using defaults.")
            self.settings = {
                "history_window": {"intent_recognition": 3, "slot_extraction": 5, "response_generation": 5},
                "safety": {"confirm_unsafe_methods": True}
            }

    def process_message(self, user_message: str) -> str:
        logger.info(f"User Message: {user_message}")
        
        self.history.append({"role": "user", "content": user_message})

        # 0. Handle Confirmation
        if self.waiting_for_confirmation:
            if "yes" in user_message.lower() or "confirm" in user_message.lower():
                self.waiting_for_confirmation = False
                result = self._execute_tool()
                self.history.append({"role": "assistant", "content": result})
                return result
            else:
                self.waiting_for_confirmation = False
                self.current_tool = None
                self.state = {}
                response = "Action cancelled."
                self.history.append({"role": "assistant", "content": response})
                return response
        
        # 1. Identify Intent (if no active tool)
        if not self.current_tool:
            tool_name = self._identify_intent()
            if tool_name:
                self.current_tool = self.registry.get_tool(tool_name)
                self.state = {p: None for p in self.current_tool['parameters'].keys()}
                logger.info(f"Selected Tool: {tool_name}")
            else:
                response = "I'm sorry, I didn't understand that request. I can help with updating nominees, filing claims, or checking policy details."
                self.history.append({"role": "assistant", "content": response})
                return response

        # 2. Extract Slots
        self._extract_slots()

        # 3. Check for Missing Slots
        missing = [p for p, v in self.state.items() 
                   if v is None and self.current_tool['parameters'][p]['required']]
        
        if missing:
            response = self._generate_response(missing)
            self.history.append({"role": "assistant", "content": response})
            return response
        
        # 4. Check Safety (Confirmation) - Logic controlled by config
        if self.settings['safety'].get('confirm_unsafe_methods', True):
            method = self.current_tool.get('method', 'POST')
            if method in ['POST', 'PUT', 'DELETE']:
                self.waiting_for_confirmation = True
                summary = json.dumps({k: v for k, v in self.state.items() if v})
                response = f"I am about to execute a {method} request with: {summary}. Are you sure? (Yes/No)"
                self.history.append({"role": "assistant", "content": response})
                return response

        # 5. Execute (Safe GET or if checks passed)
        result = self._execute_tool()
        self.history.append({"role": "assistant", "content": result})
        return result

    def _generate_response(self, missing_slots: List[str]) -> str:
        filled = {k: v for k, v in self.state.items() if v is not None}
        
        # Smart Context: Inject Date
        from datetime import datetime
        current_date = datetime.now().strftime("%Y-%m-%d")
        
        # Load Prompt Template
        template = self.settings['prompts'].get('response_system', "")
        system_content = template.format(
            tool_name=self.current_tool['name'],
            current_date=current_date,
            filled_data=json.dumps(filled),
            missing_slots=json.dumps(missing_slots)
        )
        
        window = self.settings['history_window'].get('response_generation', 5)
        messages = [{"role": "system", "content": system_content}] + self.history[-window:] # Configurable window
        
        # Call LLM
        raw_response = self.llm.predict(messages)
        
        # Parse Thought vs Response
        import re
        thought_match = re.search(r'<thought>(.*?)</thought>', raw_response, re.DOTALL)
        thought = thought_match.group(1).strip() if thought_match else "Thinking..."
        
        # Clean response (remove tags)
        final_response = re.sub(r'<thought>.*?</thought>', '', raw_response, flags=re.DOTALL).strip()
        
        return f"[THOUGHT] {thought}\n[RESPONSE] {final_response}"

    def _identify_intent(self) -> str:
        # Load Prompt Template
        template = self.settings['prompts'].get('intent_system', "")
        system_content = template.format(
            tool_names=", ".join(self.registry.tools.keys())
        )
        
        window = self.settings['history_window'].get('intent_recognition', 3)
        messages = [{"role": "system", "content": system_content}] + self.history[-window:]
        
        response = self.llm.predict(messages)
        logger.info(f"Intent Raw Response: {response}")
        try:
            data = json.loads(response)
            tool_name = data.get("tool")
            if tool_name and self.registry.get_tool(tool_name):
                return tool_name
            else:
                logger.warning(f"LLM returned invalid tool: {tool_name}")
                return None
        except Exception as e:
            logger.error(f"Failed to parse intent: {e}")
            return None

    def _extract_slots(self):
        # Smart Context: Inject Date for relative date resolution (e.g., "yesterday")
        from datetime import datetime
        current_date = datetime.now().strftime("%Y-%m-%d")
        
        # Load Prompt Template
        template = self.settings['prompts'].get('extraction_system', "")
        system_content = template.format(
            slots=list(self.state.keys()),
            current_date=current_date
        )
        
        window = self.settings['history_window'].get('slot_extraction', 5)
        messages = [{"role": "system", "content": system_content}] + self.history[-window:]
        
        response = self.llm.predict(messages)
        try:
            data = json.loads(response)
            for k, v in data.items():
                if k in self.state and v:
                    self.state[k] = v
                    logger.info(f"Extracted {k}: {v}")
        except:
            pass

    def _execute_tool(self) -> str:
        tool_name = self.current_tool['name']
        logger.info(f"Executing {tool_name} with {self.state}")
        
        # Production Logic:
        # 1. Get endpoint and method from config
        endpoint = self.current_tool.get('endpoint')
        method = self.current_tool.get('method', 'POST')
        
        # 2. Construct the Request
        # In a real app, you would use 'requests' here.
        # For this demo, we will simulate the HTTP call log to show HOW it works.
        
        import requests
        
        # Handling path parameters (e.g. /policy/{policy_number})
        final_url = endpoint
        json_payload = {}
        
        for key, value in self.state.items():
            placeholder = "{" + key + "}"
            if placeholder in final_url:
                final_url = final_url.replace(placeholder, str(value))
            else:
                json_payload[key] = value

        curl_command = f"curl -X {method} '{final_url}' -H 'Content-Type: application/json' -d '{json.dumps(json_payload)}'"
        
        # In a real production system, this is the line:
        # try:
        #     if method == "GET":
        #         resp = requests.get(final_url, params=json_payload)
        #     else:
        #         resp = requests.post(final_url, json=json_payload)
        #     return f"API Success: {resp.json()}"
        # except Exception as e:
        #     return f"API Failure: {e}"

        # Reset state after execution
        self.current_tool = None
        self.state = {}
        
        return f"Executing Production API Call:\n> {curl_command}\n\n(Simulated Success)"
