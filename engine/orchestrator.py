"""
Orchestrator - Core agent engine using Strategy Pattern.

Strategies are plug-and-play. To remove a feature, just comment out the import:
    # from engine.strategies import PlannerStrategy
"""
import json
from typing import Dict, Any, List
from engine.registry import ToolRegistry
from engine.llm import LLMService
from utils.logger import setup_logger

# Import strategies (comment out to disable features)
from engine.strategies.fast_path import FastPathStrategy
from engine.strategies.planner import PlannerStrategy
from engine.strategies.react import ReActStrategy

logger = setup_logger("Orchestrator")


class Orchestrator:
    def __init__(self, registry: ToolRegistry, llm: LLMService):
        self.registry = registry
        self.llm = llm
        self.sessions = {}
        
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
                "safety": {"confirm_unsafe_methods": True},
                "prompts": {}
            }
        
        # Initialize strategies (order matters - first match wins)
        self.strategies = [
            FastPathStrategy(self),   # Try simple path first
            PlannerStrategy(self),    # Then try batch planning
            ReActStrategy(self),      # Fallback to full reasoning
        ]

    def process_message(self, user_message: str, session_id: str = "default") -> str:
        """Main entry point. Dispatches to appropriate strategy."""
        # Load/Create Session
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                "history": [],
                "observations": [],
                "user_goal": "",
                "current_tool": None,
                "state": {},
                "waiting_for_confirmation": False,
                "pending_plan": None,
                "executing_plan": False
            }
        
        session = self.sessions[session_id]
        logger.info(f"Processing message for Session: {session_id}")
        
        session["user_goal"] = user_message
        session["history"].append({"role": "user", "content": user_message})
        session["observations"] = []
        
        # Handle pending confirmation
        if session["waiting_for_confirmation"]:
            return self._handle_confirmation(session, user_message)
        
        # Handle pending plan confirmation
        if session.get("pending_plan"):
            return self._handle_plan_confirmation(session, user_message)
        
        # Dispatch to strategies
        for strategy in self.strategies:
            if strategy.can_handle(session, user_message):
                result = strategy.process(session, user_message)
                if result:
                    return result
        
        # No strategy handled it
        response = "I'm not sure how to help with that."
        session["history"].append({"role": "assistant", "content": response})
        return response

    # === Core Utilities (used by all strategies) ===
    
    def _handle_confirmation(self, session: dict, user_message: str) -> str:
        """Handle user confirmation for unsafe operations."""
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

    def _handle_plan_confirmation(self, session: dict, user_message: str) -> str:
        """Handle batch plan confirmation."""
        planner = next((s for s in self.strategies if isinstance(s, PlannerStrategy)), None)
        if planner and "yes" in user_message.lower():
            return planner.execute_plan(session)
        else:
            session["pending_plan"] = None
            response = "Plan cancelled."
            session["history"].append({"role": "assistant", "content": response})
            return response

    def _needs_confirmation(self, session: dict) -> bool:
        """Check if the current action needs user confirmation."""
        if session.get("executing_plan"):
            return False  # Skip confirmation during batch execution
        if not self.settings['safety'].get('confirm_unsafe_methods', True):
            return False
        method = session["current_tool"].get('method', 'POST')
        unsafe = self.settings['safety'].get('unsafe_methods', ['POST', 'PUT', 'DELETE'])
        return method in unsafe

    def _generate_response(self, session: dict, missing_slots: List[str]) -> str:
        """Generate a response asking for missing information."""
        filled = {k: v for k, v in session["state"].items() if v is not None}
        
        from datetime import datetime
        current_date = datetime.now().strftime("%Y-%m-%d")
        
        template = self.settings['prompts'].get('response_system', "")
        system_content = template.format(
            tool_name=session["current_tool"]['name'],
            current_date=current_date,
            filled_data=json.dumps(filled),
            missing_slots=json.dumps(missing_slots)
        )
        
        messages = [{"role": "system", "content": system_content}] + session["history"][-5:]
        raw_response = self.llm.predict(messages)
        
        import re
        thought_match = re.search(r'<thought>(.*?)</thought>', raw_response, re.DOTALL)
        thought = thought_match.group(1).strip() if thought_match else "Thinking..."
        final_response = re.sub(r'<thought>.*?</thought>', '', raw_response, flags=re.DOTALL).strip()
        
        return f"[THOUGHT] {thought}\n[RESPONSE] {final_response}"

    def _extract_slots(self, session: dict):
        """Extract slot values from conversation history."""
        from datetime import datetime
        current_date = datetime.now().strftime("%Y-%m-%d")
        
        template = self.settings['prompts'].get('extraction_system', "")
        system_content = template.format(
            slots=list(session["state"].keys()),
            current_date=current_date
        )
        
        messages = [{"role": "system", "content": system_content}] + session["history"][-5:]
        
        response = self.llm.predict(messages)
        try:
            data = json.loads(response)
            for k, v in data.items():
                if k in session["state"] and v:
                    session["state"][k] = v
                    logger.info(f"Extracted {k}: {v}")
        except:
            pass

    def _execute_tool(self, session: dict) -> str:
        """Execute the current tool and return the result."""
        tool = session["current_tool"]
        tool_name = tool['name']
        logger.info(f"Executing {tool_name} with {session['state']}")
        
        endpoint = tool.get('endpoint', '')
        method = tool.get('method', 'POST')
        
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
        
        mock_response = tool.get("mock_response")
        
        session["current_tool"] = None
        session["state"] = {}
        
        if mock_response:
            return mock_response
            
        return f"API Call: {curl_command}\n(Simulated Success)"
