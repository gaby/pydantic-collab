"""
Example 10: Handoff including previous conversation

This verifies that when an agent returns a HandoffOutput with
`include_conversation=True`, the next agent receives the previous
message history as part of its prompt builder context.
"""
import asyncio

import logfire
from pydantic_ai import Agent

from pydantic_collab import Collab, CollabAgent, StarCollab
from example_tools import generic_tool

logfire.configure()
logfire.instrument_pydantic_ai()
logfire.instrument_httpx(capture_all=True)
MODEL = "gemini-2.0-flash"


def create_swarm():
    lead = Agent(MODEL, name="Lead", system_prompt="You decide whether to escalate.")
    expert = Agent(MODEL, name="Expert", system_prompt="You are an expert who needs previous convo.")

    swarm = StarCollab(
        agents=[
            CollabAgent(agent=lead, description="Lead", agent_calls=("Expert",)),
            CollabAgent(agent=expert, description="Expert", agent_calls=()),
        ],
        max_handoffs=3,
        tools=(generic_tool,)
    )
    return swarm


async def main():
    swarm = create_swarm()
    # Run a query that should cause Lead to hand off and include conversation
    query = "Please evaluate if this requires escalation: user reports data loss during sync"
    res = await swarm.run(query)
    print("Final output:", res.output)
    print("Execution history:")
    print(res.print_execution_flow())


if __name__ == '__main__':
    asyncio.run(main())
