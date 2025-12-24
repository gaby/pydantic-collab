"""
Example 3: Parallel Fan-Out with Specialists
=============================================

Topology: Coordinator (center) can call Technical and Creative specialists

The Coordinator routes queries to specialists based on content type
and synthesizes their responses.

Use case: Multi-perspective analysis, parallel specialist consultation
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
    """Create swarm with fan-out to specialists."""

    coordinator = Agent(
        MODEL,
        name="Coordinator",
        system_prompt="""You are a coordinating agent. Analyze the query and:
- Call "Technical" agent for code, engineering, data questions
- Call "Creative" agent for design, writing, artistic questions

You can call multiple specialists if needed.
Synthesize their responses into a comprehensive final answer.""",
    )

    technical = Agent(
        MODEL,
        name="Technical",
        system_prompt="""You are a technical specialist.
Provide technical analysis, solutions, and implementation details.
Focus on code, engineering, and data-related aspects.""",
    )

    creative = Agent(
        MODEL,
        name="Creative",
        system_prompt="""You are a creative specialist.
Provide creative ideas, artistic perspectives, and design solutions.
Focus on aesthetics, user experience, and innovative approaches.""",
    )

    # Star topology: Coordinator can call both specialists
    swarm = StarCollab(
        agents=[
            CollabAgent(
                agent=coordinator,
                description="Coordinator that routes to specialists",
                agent_calls=("Technical", "Creative"),
            ),
            CollabAgent(
                agent=technical,
                description="Technical specialist for engineering questions",
                agent_calls=(),
            ),
            CollabAgent(
                agent=creative,
                description="Creative specialist for design questions",
                agent_calls=(),
            ),
        ],
        max_handoffs=5,
    )

    return swarm


async def main():
    """Run the example."""
    swarm = create_swarm()

    print("=== Parallel Fan-Out Example ===")
    print("Topology: Coordinator → (Technical, Creative)")
    print()

    # Test different query types
    queries = [
        "How do I optimize a database query?",  # Technical
        "Design a logo for my bakery",  # Creative
    ]

    for query in queries:
        print(f"Query: {query}")
        print("-" * 40)

        result = await swarm.run(query)

        print(f"Response: {result.output}")
        print(f"Path: {' → '.join(result.execution_path)}")
        print(f"Iterations: {result.iterations}")
        print()


if __name__ == "__main__":
    asyncio.run(main())
