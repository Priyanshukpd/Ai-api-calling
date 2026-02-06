import json
from typing import Optional
from engine.strategies.base import BaseStrategy
from utils.logger import setup_logger

logger = setup_logger("ReAct")

MAX_ITERATIONS = 5

class ReActStrategy(BaseStrategy):
    """
    ReAct (Reasoning + Acting) strategy.
    THINK -> ACT -> OBSERVE -> REFLECT loop for complex reasoning.
    """
    
    def can_handle(self, session: dict, message: str) -> bool:
        """ReAct is the fallback - always can handle."""
        return True
    
    def process(self, session: dict, message: str) -> Optional[str]:
        """Execute the ReAct loop."""
        logger.info("Starting ReAct loop")
        
        for iteration in range(MAX_ITERATIONS):
            logger.info(f"=== ReAct Iteration {iteration + 1} ===")
            
            # THINK
            decision = self._think(session)
            logger.info(f"THINK: {decision}")
            
            # Check confidence
            confidence = decision.get("confidence", 1.0)
            if confidence < 0.7 and decision["action"] == "use_tool":
                tool_name = decision.get("tool", "unknown")
                reasoning = decision.get("reasoning", "")
                response = f"[LOW CONFIDENCE: {confidence:.0%}] I think you want to use '{tool_name}' ({reasoning}). Is that correct?"
                session["history"].append({"role": "assistant", "content": response})
                return response
            
            if decision["action"] == "respond":
                response = decision.get("message", "I'm not sure how to help with that.")
                session["history"].append({"role": "assistant", "content": response})
                return response
            
            elif decision["action"] == "ask_user":
                response = decision.get("message", "Could you clarify?")
                session["history"].append({"role": "assistant", "content": response})
                return response
            
            elif decision["action"] == "use_tool":
                tool_name = decision.get("tool")
                if not tool_name:
                    continue
                
                tool = self.registry.get_tool(tool_name)
                if not tool:
                    logger.warning(f"Tool not found: {tool_name}")
                    continue
                
                session["current_tool"] = tool
                session["state"] = {p: None for p in tool['parameters'].keys()}
                
                self.orchestrator._extract_slots(session)
                
                missing = [p for p, v in session["state"].items() 
                           if v is None and tool['parameters'][p]['required']]
                
                if missing:
                    response = self.orchestrator._generate_response(session, missing)
                    session["history"].append({"role": "assistant", "content": response})
                    return response
                
                if self.orchestrator._needs_confirmation(session):
                    session["waiting_for_confirmation"] = True
                    summary = json.dumps({k: v for k, v in session["state"].items() if v})
                    response = f"Ready to execute {tool_name} with: {summary}. Confirm? (Yes/No)"
                    session["history"].append({"role": "assistant", "content": response})
                    return response
                
                # ACT
                result = self.orchestrator._execute_tool(session)
                
                # OBSERVE
                session["observations"].append({
                    "tool": tool_name,
                    "result": result
                })
                logger.info(f"OBSERVE: {tool_name} -> {result[:100]}...")
                
                # REFLECT
                reflection = self._reflect(session, tool_name, result)
                logger.info(f"REFLECT: {reflection}")
                
                if reflection.get("complete", True):
                    final_response = f"[ACTION] Executed {tool_name}\n[RESULT] {result}"
                    session["history"].append({"role": "assistant", "content": final_response})
                    return final_response
                else:
                    logger.info(f"Continuing loop: {reflection.get('next_step', 'unknown')}")
                    continue
        
        response = "I've tried multiple approaches but couldn't complete your request."
        session["history"].append({"role": "assistant", "content": response})
        return response
    
    def _think(self, session: dict) -> dict:
        """THINK phase: Decide what action to take."""
        template = self.settings['prompts'].get('think_system', "")
        
        observations_str = json.dumps(session["observations"]) if session["observations"] else "None yet"
        
        system_content = template.format(
            tool_names=", ".join(self.registry.tools.keys()),
            observations=observations_str
        )
        
        messages = [{"role": "system", "content": system_content}] + session["history"][-5:]
        
        response = self.llm.predict(messages)
        try:
            return json.loads(response)
        except:
            logger.error(f"Failed to parse THINK: {response}")
            return {"action": "respond", "message": response}
    
    def _reflect(self, session: dict, tool_name: str, tool_result: str) -> dict:
        """REFLECT phase: Decide if goal is achieved."""
        template = self.settings['prompts'].get('reflect_system', "")
        
        system_content = template.format(
            tool_name=tool_name,
            tool_result=tool_result[:500],
            user_goal=session["user_goal"]
        )
        
        messages = [{"role": "system", "content": system_content}]
        
        response = self.llm.predict(messages)
        try:
            return json.loads(response)
        except:
            logger.error(f"Failed to parse REFLECT: {response}")
            return {"complete": True}
