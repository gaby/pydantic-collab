"""Example 2: Star Topology with Expert Consultation
==================================================

Topology: Assistant (center) can call Expert as a tool

The Assistant handles user queries and can consult the Expert
for technical questions by calling it as a tool.

Use case: Support with specialist consultation
"""

import asyncio

import logfire
from pydantic_ai import Agent

from pydantic_collab import Collab, CollabAgent, StarCollab

logfire.configure()
logfire.instrument_pydantic_ai()
logfire.instrument_httpx(capture_all=True)

MODEL = "google-gla:gemini-2.0-flash"


def create_swarm():
    """Create swarm with star topology - Assistant calls Expert."""

    assistant = Agent(
        MODEL,
        name="Assistant",
        system_prompt="""You are a helpful assistant. Your job is to:
1. Try to answer user questions directly
2. If the question is too technical, call the Expert agent for help
3. Incorporate Expert's response into your final answer

You can call the Expert agent using the call_agent tool when you need technical expertise.""",
    )

    expert = Agent(
        MODEL,
        name="Expert",
        system_prompt="""You are a technical expert. Your job is to:
1. Provide deep technical expertise when consulted
2. Give detailed, accurate technical information
3. Return your expertise to the calling agent""",
    )

    # Star topology: Assistant is the starting/final agent and can call Expert
    swarm = StarCollab(
        agents=[
            CollabAgent(
                agent=assistant,
                description='Helpful assistant that can consult experts',
                agent_calls=('Expert',),  # Assistant can call Expert as a tool
            ),
            CollabAgent(
                agent=expert,
                description='Technical expert for deep technical questions',
                agent_calls=(),
            ),
        ],
    )

    return swarm


async def main():
    """Run the example."""
    swarm = create_swarm()

    print('=== Star Topology Example ===')
    print('Topology: Assistant (center) can call Expert as tool')
    print()

    # Test queries
    queries = [
        "What's the weather like?",  # Simple - Assistant handles
        'Explain quantum entanglement',  # Technical - needs Expert
    ]

    for query in queries:
        print(f'Query: {query}')
        print("-" * 40)

        result = await swarm.run(query)

        print(f'Response: {result.output}')
        print(f"Path: {' → '.join(result.execution_path)}")
        print(f'Iterations: {result.iterations}')
        print()


if __name__ == '__main__':
    asyncio.run(main())
