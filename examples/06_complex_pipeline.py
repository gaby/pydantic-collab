"""Example 6: Content Creation Pipeline
=====================================

Topology: Intake → Researcher → Writer → Publisher (forward handoff chain)

A content creation pipeline where each stage processes and passes
to the next for a complete workflow.

Use case: Content creation pipeline, document processing, code review workflow
"""

import asyncio

import logfire
from pydantic_ai import Agent

from pydantic_collab import CollabAgent, PiplineCollab

logfire.configure()
logfire.instrument_pydantic_ai()
logfire.instrument_httpx(capture_all=True)

MODEL = 'google-gla:gemini-2.0-flash'


def create_swarm():
    """Create content pipeline swarm."""
    intake = Agent(
        MODEL,
        name='Intake',
        system_prompt="""You are the intake agent for a content pipeline.
1. Receive content requests
2. Clarify the requirements
3. Summarize what content is needed
4. Hand off to Researcher for fact-finding""",
    )

    researcher = Agent(
        MODEL,
        name='Researcher',
        system_prompt="""You are a researcher. Your job is to:
1. Receive content requirements from Intake
2. Gather background information and facts
3. Identify key points to cover
4. Hand off findings to Writer""",
    )

    writer = Agent(
        MODEL,
        name='Writer',
        system_prompt="""You are the content writer. Your job is to:
1. Receive research from Researcher
2. Create polished, engaging content
3. Structure the content clearly
4. Hand off to Publisher for final formatting""",
    )

    publisher = Agent(
        MODEL,
        name='Publisher',
        system_prompt="""You are the publisher. Your job is to:
1. Receive draft content from Writer
2. Format for publication
3. Add any final polish
4. Return the final published content to the user""",
    )

    # Forward handoff chain: Intake → Researcher → Writer → Publisher
    swarm = PiplineCollab(
        agents=[
            CollabAgent(agent=intake, description='Intake - receives and clarifies content requests'),
            CollabAgent(agent=researcher, description='Researcher - gathers facts and background'),
            CollabAgent(agent=writer, description='Writer - creates polished content'),
            CollabAgent(agent=publisher, description='Publisher - formats and publishes final content'),
        ],
        max_handoffs=10,
    )

    return swarm


async def main():
    """Run the example."""
    swarm = create_swarm()

    print('=== Content Creation Pipeline Example ===')
    print('Topology: Intake → Researcher → Writer → Publisher')
    print()

    query = 'Write a blog post about the future of AI in healthcare'
    print(f'Query: {query}')
    print('-' * 60)

    result = await swarm.run(query)

    print(f'Response: {result.output}')
    print(f'Final agent: {result.final_agent}')
    print(f'Iterations: {result.iterations}')
    print(f'Path: {" → ".join(result.execution_path)}')
    print()
    print(result.print_execution_flow())


if __name__ == '__main__':
    asyncio.run(main())
