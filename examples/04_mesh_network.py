"""
Example 4: Mesh Network - Full Collaboration
=============================================

Topology: Strategist ↔ Technologist ↔ Designer (all connected)

All agents can communicate with each other as tools.
Each brings a unique perspective and they collaborate freely.

Use case: Brainstorming, multi-disciplinary problem solving
"""

import asyncio

import logfire

from pydantic_collab import Collab, CollabAgent, MeshCollab

logfire.configure()
logfire.instrument_pydantic_ai()
logfire.instrument_httpx(capture_all=True)

MODEL = "google-gla:gemini-2.0-flash"


def create_swarm():
    """Create swarm with mesh topology."""

    # Mesh topology: starting agent can call all others, others can call each other
    swarm = MeshCollab(
        agents=[
            CollabAgent(
                MODEL,
                name="Strategist",
                system_prompt="""You are a business strategist in a collaborative team.

Your teammates (call them as tools when needed):
- Technologist: Knows implementation details
- Designer: Focuses on user experience

Collaborate to solve problems. Use call_agent to consult teammates.
Provide a comprehensive strategic recommendation when ready.""",
                agent_calls=("Technologist", "Designer"),
            ),
            CollabAgent(
                MODEL,
                name="Technologist",
                system_prompt="""You are a technologist in a collaborative team.

Your teammates (call them as tools when needed):
- Strategist: Handles business strategy
- Designer: Focuses on user experience

Provide technical insights and implementation feasibility.
Call teammates for their perspectives when helpful.""",
                agent_calls=("Strategist", "Designer"),
            ),
            CollabAgent(
                MODEL,
                name="Designer",
                system_prompt="""You are a UX designer in a collaborative team.

Your teammates (call them as tools when needed):
- Strategist: Handles business strategy
- Technologist: Knows implementation details

Focus on user experience and usability.
Call teammates for their perspectives when helpful.""",
                agent_calls=("Strategist", "Technologist"),
            ),
        ],
        max_handoffs=8,  # Allow collaboration but prevent infinite loops
    )

    return swarm


async def main():
    """Run the example."""
    swarm = create_swarm()

    print("=== Mesh Network Example ===")
    print("Topology: Strategist ↔ Technologist ↔ Designer (all connected)")
    print()

    query = "We need to build a mobile app for tracking personal finances. What should we consider?"
    print(f"Query: {query}")
    print("-" * 50)

    result = await swarm.run(query)

    print(f"Response: {result.output}")
    print(f"Final agent: {result.final_agent}")
    print(f"Path: {' → '.join(result.execution_path)}")
    print(f"Iterations: {result.iterations}")
    print()
    print(result.print_execution_flow())


if __name__ == "__main__":
    asyncio.run(main())
