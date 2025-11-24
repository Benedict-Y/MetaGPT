import asyncio
from metagpt.roles import Role
from metagpt.schema import Message
from metagpt.actions import Action
from metagpt.configs.llm_config import LLMConfig
from metagpt.llm import LLM

# --- Define Actions ---

class CoFReasoning(Action):
    """Action for Chain-of-Focus reasoning (Planner/Reasoning Core)"""
    name: str = "CoFReasoning"
    
    async def run(self, instruction: str):
        prompt = f"You are a video reasoning expert. Analyze this request: {instruction}"
        # This action will use the LLM configured for the Role
        rsp = await self.llm.aask(prompt)
        return rsp

class OpenO3Description(Action):
    """Action for Open-o3-Video description (Tool Agent)"""
    name: str = "OpenO3Description"

    async def run(self, instruction: str):
        prompt = f"You are a video description expert. Describe the visual content for: {instruction}"
        # This action will use the LLM configured for the Role
        rsp = await self.llm.aask(prompt)
        return rsp

# --- Define Roles ---

class CoFPlanner(Role):
    name: str = "CoFPlanner"
    profile: str = "Planner"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.set_actions([CoFReasoning])

class OpenO3Agent(Role):
    name: str = "OpenO3Agent"
    profile: str = "ToolAgent"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.set_actions([OpenO3Description])

# --- Main Interaction Script ---

async def main():
    # 1. Configure CoF Model (Port 8005)
    cof_llm_config = LLMConfig(
        base_url="http://localhost:8005/v1",
        model="CoF-rl-model-7b",
        api_key="EMPTY"
    )

    # 2. Configure Open-o3 Model (Port 8006)
    openo3_llm_config = LLMConfig(
        base_url="http://localhost:8006/v1",
        model="Open-o3-Video",
        api_key="EMPTY"
    )

    # 3. Initialize Roles with specific LLM configs
    
    planner = CoFPlanner()
    planner.llm = LLM(llm_config=cof_llm_config) # Force specific LLM
    planner._watch([Message]) # Watch for user messages

    tool_agent = OpenO3Agent()
    tool_agent.llm = LLM(llm_config=openo3_llm_config) # Force specific LLM
    tool_agent._watch([CoFReasoning]) # Watch for Planner's output

    # 4. Simulate Interaction
    user_msg = "Analyze the motion of a person jumping in the video."
    print(f"\n[User]: {user_msg}")

    # Step 1: Planner (CoF) analyzes the request
    print(f"\n[Planner ({cof_llm_config.model})] is thinking...")
    
    # Use planner.run() instead of accessing rc.todo directly
    # planner.run() handles _observe -> _think -> _act
    planner_msg = await planner.run(user_msg)
    
    # Check if planner_msg is None (which can happen if _think returns False or no action taken)
    if planner_msg is None:
        print("[Planner Output]: No response generated (Planner decided not to act).")
        return

    planner_response = planner_msg.content
    print(f"[Planner Output]: {planner_response}")

    # Step 2: Tool Agent (Open-o3) acts on Planner's output
    print(f"\n[Tool Agent ({openo3_llm_config.model})] is working...")
    
    # Use tool_agent.run()
    tool_msg = await tool_agent.run(planner_response)
    
    if tool_msg is None:
        print("[Tool Agent Output]: No response generated.")
        return

    tool_response = tool_msg.content
    print(f"[Tool Agent Output]: {tool_response}")

if __name__ == "__main__":
    asyncio.run(main())
