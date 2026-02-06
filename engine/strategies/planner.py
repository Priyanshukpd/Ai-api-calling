import json
from typing import Optional
from engine.strategies.base import BaseStrategy
from utils.logger import setup_logger

logger = setup_logger("Planner")

class PlannerStrategy(BaseStrategy):
    """
    Batch confirm strategy for complex multi-tool requests.
    Creates a plan upfront and executes all steps after single confirmation.
    """
    
    COMPLEXITY_MARKERS = [" and ", " then ", " also ", " after that", "first ", "second "]
    
    def can_handle(self, session: dict, message: str) -> bool:
        """Return True if this is a multi-step request."""
        is_complex = any(marker in message.lower() for marker in self.COMPLEXITY_MARKERS)
        return is_complex
    
    def process(self, session: dict, message: str) -> Optional[str]:
        """Create a plan and ask for confirmation."""
        plan = self._create_plan(session, message)
        
        if plan and len(plan.get("steps", [])) > 1:
            session["pending_plan"] = plan
            steps_display = "\n".join([f"  {i+1}. {s['tool']}: {s['purpose']}" 
                                        for i, s in enumerate(plan["steps"])])
            response = f"[PLAN] I'll execute these steps:\n{steps_display}\n\nConfirm all? (Yes/No)"
            session["history"].append({"role": "assistant", "content": response})
            return response
        
        # Fall through to ReAct if planning failed
        return None
    
    def _create_plan(self, session: dict, message: str) -> dict:
        """Create a multi-step plan using LLM."""
        template = self.settings['prompts'].get('plan_system', "")
        if not template:
            return None
            
        system_content = template.format(
            tool_names=", ".join(self.registry.tools.keys())
        )
        
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": message}
        ]
        
        response = self.llm.predict(messages)
        try:
            plan = json.loads(response)
            logger.info(f"Created plan with {len(plan.get('steps', []))} steps")
            return plan
        except:
            logger.error(f"Failed to parse plan: {response}")
            return None
    
    def execute_plan(self, session: dict) -> str:
        """Execute a confirmed plan."""
        plan = session["pending_plan"]
        session["pending_plan"] = None
        session["executing_plan"] = True
        
        logger.info(f"Executing plan with {len(plan['steps'])} steps")
        
        results = []
        for i, step in enumerate(plan["steps"]):
            tool_name = step["tool"]
            tool = self.registry.get_tool(tool_name)
            
            if not tool:
                results.append(f"Step {i+1}: Tool '{tool_name}' not found")
                continue
            
            session["current_tool"] = tool
            session["state"] = {p: None for p in tool['parameters'].keys()}
            
            self.orchestrator._extract_slots(session)
            
            missing = [p for p, v in session["state"].items() 
                       if v is None and tool['parameters'][p]['required']]
            
            if missing:
                results.append(f"Step {i+1} ({tool_name}): Missing {missing} - skipped")
                session["current_tool"] = None
                session["state"] = {}
                continue
            
            result = self.orchestrator._execute_tool(session)
            results.append(f"Step {i+1} ({tool_name}): {result}")
            
            session["observations"].append({
                "tool": tool_name,
                "result": result
            })
        
        session["executing_plan"] = False
        
        response = "[PLAN EXECUTED]\n" + "\n".join(results)
        session["history"].append({"role": "assistant", "content": response})
        return response
