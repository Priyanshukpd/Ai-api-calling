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
        self.sessions = {} # Dictionary to store state/history per session_id
        
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

    def process_message(self, user_message: str, session_id: str = "default") -> str:
        # 1. Load Session State
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                "history": [],
                "state": {},
                "current_tool": None,
                "waiting_for_confirmation": False
            }
        
        session = self.sessions[session_id]
        logger.info(f"Processing message for Session ID: {session_id}")
        logger.info(f"User Message: {user_message}")
        
        session["history"].append({"role": "user", "content": user_message})

        # 0. Handle Confirmation
        if session["waiting_for_confirmation"]:
            if "yes" in user_message.lower() or "confirm" in user_message.lower():
                session["waiting_for_confirmation"] = False
                result = self._execute_tool(session)
                session["history"].append({"role": "assistant", "content": result})
                return result
            else:
                session["waiting_for_confirmation"] = False
                session["current_tool"] = None
                session["state"] = {}
                response = "Action cancelled."
                session["history"].append({"role": "assistant", "content": response})
                return response
        
        # 1. Identify Intent (if no active tool)
        if not session["current_tool"]:
            tool_name = self._identify_intent(session)
            if tool_name:
                session["current_tool"] = self.registry.get_tool(tool_name)
                # Initialize slots for the tool
                session["state"] = {p: None for p in session["current_tool"]['parameters'].keys()}
                logger.info(f"Selected Tool: {tool_name}")
            else:
                response = "I'm sorry, I didn't understand that request. I can help with updating nominees, filing claims, or checking policy details."
                session["history"].append({"role": "assistant", "content": response})
                return response

        # 2. Extract Slots
        self._extract_slots(session)

        # 3. Check for Missing Slots
        missing = [p for p, v in session["state"].items() 
                   if v is None and session["current_tool"]['parameters'][p]['required']]
        
        if missing:
            response = self._generate_response(session, missing)
            session["history"].append({"role": "assistant", "content": response})
            return response
        
        # 4. Check Safety (Confirmation) - Logic controlled by config
        if self.settings['safety'].get('confirm_unsafe_methods', True):
            method = session["current_tool"].get('method', 'POST')
            # Check config for unsafe methods list, default to POST/PUT/DELETE
            unsafe = self.settings['safety'].get('unsafe_methods', ['POST', 'PUT', 'DELETE'])
            if method in unsafe:
                session["waiting_for_confirmation"] = True
                summary = json.dumps({k: v for k, v in session["state"].items() if v})
                response = f"I am about to execute a {method} request with: {summary}. Are you sure? (Yes/No)"
                session["history"].append({"role": "assistant", "content": response})
                return response

        # 5. Execute (Safe calls or if passed checks)
        result = self._execute_tool(session)
        session["history"].append({"role": "assistant", "content": result})
        return result

    def _generate_response(self, session, missing_slots: List[str]) -> str:
        filled = {k: v for k, v in session["state"].items() if v is not None}
        
        # Smart Context: Inject Date
        from datetime import datetime
        current_date = datetime.now().strftime("%Y-%m-%d")
        
        # Load Prompt Template
        template = self.settings['prompts'].get('response_system', "")
        system_content = template.format(
            tool_name=session["current_tool"]['name'],
            current_date=current_date,
            filled_data=json.dumps(filled),
            missing_slots=json.dumps(missing_slots)
        )
        
        window = self.settings['history_window'].get('response_generation', 5)
        messages = [{"role": "system", "content": system_content}] + session["history"][-window:] 
        
        # Call LLM
        raw_response = self.llm.predict(messages)
        
        # Parse Thought vs Response
        import re
        thought_match = re.search(r'<thought>(.*?)</thought>', raw_response, re.DOTALL)
        thought = thought_match.group(1).strip() if thought_match else "Thinking..."
        
        # Clean response (remove tags)
        final_response = re.sub(r'<thought>.*?</thought>', '', raw_response, flags=re.DOTALL).strip()
        
        return f"[THOUGHT] {thought}\n[RESPONSE] {final_response}"

    def _identify_intent(self, session) -> str:
        # Load Prompt Template
        template = self.settings['prompts'].get('intent_system', "")
        system_content = template.format(
            tool_names=", ".join(self.registry.tools.keys())
        )
        
        window = self.settings['history_window'].get('intent_recognition', 3)
        messages = [{"role": "system", "content": system_content}] + session["history"][-window:]
        
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

    def _extract_slots(self, session):
        # Smart Context: Inject Date for relative date resolution (e.g., "yesterday")
        from datetime import datetime
        current_date = datetime.now().strftime("%Y-%m-%d")
        
        # Load Prompt Template
        template = self.settings['prompts'].get('extraction_system', "")
        system_content = template.format(
            slots=list(session["state"].keys()),
            current_date=current_date
        )
        
        window = self.settings['history_window'].get('slot_extraction', 5)
        messages = [{"role": "system", "content": system_content}] + session["history"][-window:]
        
        response = self.llm.predict(messages)
        try:
            data = json.loads(response)
            for k, v in data.items():
                if k in session["state"] and v:
                    session["state"][k] = v
                    logger.info(f"Extracted {k}: {v}")
        except:
            pass

    def _execute_tool(self, session) -> str:
        tool_name = session["current_tool"]['name']
        logger.info(f"Executing {tool_name} with {session['state']}")
        
        # Production Logic:
        # 1. Get endpoint and method from config
        endpoint = session["current_tool"].get('endpoint')
        method = session["current_tool"].get('method', 'POST')
        
        # 2. Construct the Request
        # In a real app, you would use 'requests' here.
        # For this demo, we will simulate the HTTP call log to show HOW it works.
        
        import requests
        
        # Handling path parameters (e.g. /policy/{policy_number})
        final_url = endpoint
        json_payload = {}
        
        for key, value in session["state"].items():
            placeholder = "{" + key + "}"
            if placeholder in final_url:
                final_url = final_url.replace(placeholder, str(value))
            else:
                json_payload[key] = value

        curl_command = f"curl -X {method} '{final_url}' -H 'Content-Type: application/json' -d '{json.dumps(json_payload)}'"
        
        logger.info(f"Executing: {curl_command}")
        
        # Generic Mock Response (Defined in Config)
        mock_response = None
        if "mock_response" in session["current_tool"]:
            mock_response = session["current_tool"]["mock_response"]

        # Reset state after execution
        session["current_tool"] = None
        session["state"] = {}
        
        if mock_response:
            return mock_response
            
        # Default Simulation
        return f"Executing Production API Call:\n> {curl_command}\n\n(Simulated Success)"
