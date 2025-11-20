import sys
import os
import asyncio

# Add backend to path
sys.path.append(os.path.join(os.getcwd(), "backend"))

from agent_service import AgentService

async def run_demo():
    print("Initializing Agent...")
    agent = AgentService()
    
    print("\n[SCENARIO 1] User opens the chat")
    print(f"Agent: {agent.get_welcome_message('Alex')}")
    
    print("\n[SCENARIO 2] User asks: 'Is Evolving Skies a good investment?'")
    # Passing db=None to skip Firestore write for this demo
    response = await agent.chat("Is Evolving Skies a good investment?", {"user_id": "demo_user"}, db=None)
    print(f"Agent: {response}")

if __name__ == "__main__":
    asyncio.run(run_demo())
