"""Example 9: Mixed Tools + Handoff Pipeline

Shows a pipeline where one agent uses a function tool, then hands off to another agent.
"""

import asyncio

import logfire
from example_tools import summarize_tool
from pydantic_ai import Agent

from pydantic_collab import CollabAgent, PiplineCollab

logfire.configure()
logfire.instrument_pydantic_ai()
logfire.instrument_httpx(capture_all=True)

MODEL = 'gemini-2.0-flash-lite'


def create_swarm():
    gatherer = Agent(
        MODEL,
        name='Gatherer',
        system_prompt='You gather raw info and may call tools for summarization.',
    )

    editor = Agent(
        MODEL,
        name='Editor',
        system_prompt='You refine gathered content into publishable form.',
    )

    publisher = Agent(
        MODEL,
        name='Publisher',
        system_prompt='You format and return final content.',
    )

    swarm = PiplineCollab(
        agents=[
            CollabAgent(agent=gatherer, description='Gatherer'),
            CollabAgent(agent=editor, description='Editor', agent_calls='Gatherer'),
            CollabAgent(agent=publisher, description='Publisher'),
        ],
        max_handoffs=6,
        tools=summarize_tool,
    )

    return swarm


async def main():
    swarm = create_swarm()
    query = 'Collect quotes and perspectives about AI ethics from recent papers'
    result = await swarm.run(query)
    print('Result:', result.output)
    print('Path:', ' → '.join(result.execution_path))
    print(result.print_execution_flow())


if __name__ == '__main__':
    asyncio.run(main())
