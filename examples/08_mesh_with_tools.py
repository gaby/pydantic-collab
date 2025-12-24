"""
Example 8: Mesh Topology with Tools

Demonstrates mesh connections and using a function tool (simulated search)
as a builtin tool available to agents.
"""
import asyncio

import logfire
from pydantic_ai import Agent

from pydantic_collab import Collab, CollabAgent, StarCollab
from example_tools import search_tool

logfire.configure()
logfire.instrument_pydantic_ai()
logfire.instrument_httpx(capture_all=True)

MODEL = "gemini-2.0-flash"


def create_swarm():
    strategist = Agent(
        MODEL,
        name="Strategist",
        system_prompt="You coordinate and ask specialists for input before deciding.",
    )

    technologist = Agent(
        MODEL,
        name="Technologist",
        system_prompt="You provide technical feasibility advice.",
        tools=(search_tool,),
    )

    designer = Agent(
        MODEL,
        name="Designer",
        system_prompt="You provide design considerations and UX feedback.",
        tools=(search_tool,),
    )

    # Use a star topology so only the strategist coordinates and calls specialists.
    swarm = StarCollab(
        agents=[
            CollabAgent(agent=strategist, description="Coordinator"),
            CollabAgent(agent=technologist, description="Tech expert", agent_calls=()),
            CollabAgent(agent=designer, description="Design expert", agent_calls=()),
        ],
        max_handoffs=50,
    )

    return swarm


async def main():
    swarm = create_swarm()
    # Diagnostics: print the internal topology so we can verify who can call whom
    print("Agent mapping:")
    for name, ag in swarm._name_to_agent.items():
        print(f" - {name}: {ag}")
    print("Connections:")
    for ag, conns in swarm._agent_tools.items():
        print(f" - {ag.name} can call: {[c.name for c in conns]}")
    print("Handoffs:")
    for ag, h in swarm._handoffs.items():
        print(f" - {ag.name} can handoff to: {[hh.name for hh in h]}")
    query = "Plan a cool CV for a long working technician of bridges"
    result = await swarm.run(query)
    print("Result:", result.output)
    print("Path:", " → ".join(result.execution_path))
    print(result.print_execution_flow())


if __name__ == '__main__':
    asyncio.run(main())
