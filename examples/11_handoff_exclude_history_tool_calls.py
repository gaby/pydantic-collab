"""Example 11: Handoff excluding conversation but including previous tool calls

Verifies `include_conversation=False` but `include_tool_calls_with=True` behavior.
"""

import asyncio
import os

import logfire
from example_tools import search_tool
from pydantic_ai import Agent

from pydantic_collab import CollabAgent, PiplineCollab

logfire.configure()
logfire.instrument_pydantic_ai()
logfire.instrument_httpx(capture_all=True)

# Use TestModel when no API key is present for deterministic examples
USE_TEST_MODEL = not (os.environ.get('GOOGLE_API_KEY') or os.environ.get('GEMINI_API_KEY'))
if USE_TEST_MODEL:
    from pydantic_ai.models.test import TestModel

    MODEL = TestModel()
    print('⚠️  Using TestModel for deterministic run')
else:
    MODEL = 'gemini-2.0-flash'


def create_swarm():
    front = Agent(
        MODEL,
        name='Front',
        system_prompt=(
            'You are the Front agent. If you need more backend analysis, call the builtin tool `fake_search` '
            "with the query, then return a HandoffOutput to the agent named 'Back'. "
            'When handing off, set include_conversation=False and include_tool_calls_with=True so the Back agent '
            "only receives the previous tool call traces. Be explicit: next_agent must be 'Back'."
        ),
    )
    back = Agent(MODEL, name='Back', system_prompt='You are a backend analyst, only needs tool traces.')

    swarm = PiplineCollab(
        agents=[
            # Do not register agent_calls — we want Front to call the builtin function tool only.
            CollabAgent(agent=front, description='Front'),
            CollabAgent(agent=back, description='Back'),
        ],
        max_handoffs=3,
        tools=search_tool,
    )
    return swarm


async def main():
    swarm = create_swarm()
    query = 'Investigate: intermittent failures during bulk upload'
    res = await swarm.run(query)
    print('Final output:', res.output)
    print('Execution flow:')
    print(res.print_execution_flow())


if __name__ == '__main__':
    asyncio.run(main())
