"""Example 7: Handoff Options

Demonstrates using HandoffOutput options,
include_conversation, and include_previous_handoff.
"""

import asyncio

import logfire
from pydantic_ai import Agent

from pydantic_collab import CollabAgent, PiplineCollab

logfire.configure()
logfire.instrument_pydantic_ai()
logfire.instrument_httpx(capture_all=True)

MODEL = 'gemini-2.0-flash-lite'


def create_swarm():
    intake = Agent(
        MODEL,
        name='Intake',
        system_prompt="""You are an intake agent. Summarize user request and optionally handoff.
If handing off, use HandoffOutput with next_agent set to 'Expert' or 'Support'.""",
    )

    expert = Agent(
        MODEL,
        name='Expert',
        system_prompt="""You are an expert agent. Receive a query and answer in depth or handoff back with context.""",
    )

    support = Agent(
        MODEL,
        name='Support',
        system_prompt="""You are a support agent. Provide actionable steps or ask for more info.""",
    )

    swarm = PiplineCollab(
        agents=[
            CollabAgent(agent=intake, description='Intake agent', agent_calls=('Expert', 'Support')),
            CollabAgent(agent=expert, description='Expert agent'),
            CollabAgent(agent=support, description='Support agent'),
        ],
        max_handoffs=5,
    )

    return swarm


async def main():
    swarm = create_swarm()
    query = "How can I improve my website's SEO quickly?"
    result = await swarm.run(query)
    print('Result:', result.output)
    print('Path:', ' → '.join(result.execution_path))
    print(result.print_execution_flow())


if __name__ == '__main__':
    asyncio.run(main())
