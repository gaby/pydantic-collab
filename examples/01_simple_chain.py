"""Example 1: Simple Forward Chain
===============================

Topology: Intake → Analyst → Reporter

The simplest possible swarm - a linear chain of agents.
Each agent processes and hands off to the next.

Use case: Pipeline processing (e.g., Intake → Analysis → Report)
"""

import asyncio

import logfire
from pydantic_ai import Agent

from pydantic_collab import CollabAgent, PipelineCollab

logfire.configure()
logfire.instrument_pydantic_ai()
logfire.instrument_httpx(capture_all=True)

# Use real model - requires GEMINI_API_KEY or GOOGLE_API_KEY in .env
MODEL = 'google-gla:gemini-2.0-flash'


def create_swarm():
    """Create the swarm with simple forward chain topology."""
    intake = Agent(
        MODEL,
        name='Intake',
        system_prompt="""You are an intake agent. Your job is to:
1. Understand the user's request
2. Summarize it clearly
3. Hand off to the Analyst for processing""",
    )

    analyst = Agent(
        MODEL,
        name='Analyst',
        system_prompt="""You are an analyst agent. Your job is to:
1. Receive summarized requests from Intake
2. Analyze and add insights
3. Hand off to Reporter for final output""",
    )

    reporter = Agent(
        MODEL,
        name='Reporter',
        system_prompt="""You are a reporter agent. Your job is to:
1. Receive analysis from Analyst
2. Format a clear, user-friendly response
3. Return the final response to the user""",
    )

    # Using forward_handoff: Intake → Analyst → Reporter
    swarm = PipelineCollab(
        agents=[
            CollabAgent(agent=intake, description='Intake agent - understands user requests'),
            CollabAgent(agent=analyst, description='Analyst - analyzes and adds insights'),
            CollabAgent(agent=reporter, description='Reporter - formats final response'),
        ],
        max_handoffs=5,
    )

    return swarm


async def main():
    """Run the example."""
    swarm = create_swarm()

    print('=== Simple Chain Example ===')
    print('Topology: Intake → Analyst → Reporter')
    print()

    # Test query
    query = 'I need help planning a birthday party for my daughter'
    print(f'Query: {query}')
    print()

    result = await swarm.run(query)

    print('=== Result ===')
    print(f'Response: {result.output}')
    print(f'Final agent: {result.final_agent}')
    print(f'Iterations: {result.iterations}')
    print(f'Path: {" → ".join(result.execution_path)}')
    print()
    print(result.print_execution_flow())


if __name__ == '__main__':
    asyncio.run(main())
