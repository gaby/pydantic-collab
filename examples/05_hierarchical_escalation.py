"""
Example 5: Hierarchical Support with Escalation
================================================

Topology: Coordinator (center) can call L1Support, L2Support, Escalation

A tiered support system where the Coordinator routes issues to the
appropriate support level based on complexity.

Use case: Customer support, incident management, approval workflows
"""

import asyncio

import logfire
from pydantic_ai import Agent

from pydantic_collab import CollabAgent, StarCollab

logfire.configure()
logfire.instrument_pydantic_ai()
logfire.instrument_httpx(capture_all=True)

MODEL = "google-gla:gemini-2.0-flash"


def create_swarm():
    """Create hierarchical support swarm."""

    coordinator = Agent(
        MODEL,
        name="Coordinator",
        system_prompt="""You are the support coordinator. Your job is to:
1. Assess incoming issues
2. Route to appropriate support level:
   - "L1Support": Simple, common issues (password reset, basic questions)
   - "L2Support": Complex technical issues
   - "Escalation": Critical/security issues, system outages

3. Call the appropriate support agent and synthesize their response.

Use call_agent to route to the right specialist.""",
    )

    l1_support = Agent(
        MODEL,
        name="L1Support",
        system_prompt="""You are Level 1 Support. Handle simple issues:
- Password resets
- Basic how-to questions
- Common troubleshooting

Provide clear, step-by-step solutions.
If the issue seems critical or security-related, indicate that escalation may be needed.""",
    )

    l2_support = Agent(
        MODEL,
        name="L2Support",
        system_prompt="""You are Level 2 Support. Handle complex issues:
- Technical debugging
- Configuration problems
- Integration issues

Provide detailed technical analysis and solutions.
If the issue involves a critical outage, indicate that escalation may be needed.""",
    )

    escalation = Agent(
        MODEL,
        name="Escalation",
        system_prompt="""You are the Escalation Manager. Handle critical issues:
- Security incidents
- System outages
- Urgent customer escalations

Provide a comprehensive response with:
- Immediate actions taken
- Root cause (if known)
- Follow-up steps""",
    )

    # Star topology: Coordinator routes to all support levels
    swarm = StarCollab(
        agents=[
            CollabAgent(
                agent=coordinator,
                description="Support coordinator - routes issues to appropriate level",
                agent_calls=("L1Support", "L2Support", "Escalation"),
            ),
            CollabAgent(
                agent=l1_support,
                description="Level 1 Support - handles simple issues",
                agent_calls=(),
            ),
            CollabAgent(
                agent=l2_support,
                description="Level 2 Support - handles complex technical issues",
                agent_calls=(),
            ),
            CollabAgent(
                agent=escalation,
                description="Escalation Manager - handles critical issues",
                agent_calls=(),
            ),
        ],
        max_handoffs=6,
    )

    return swarm


async def main():
    """Run the example."""
    swarm = create_swarm()

    print("=== Hierarchical Escalation Example ===")
    print("Topology: Coordinator → (L1Support, L2Support, Escalation)")
    print()

    # Test different severity levels
    queries = [
        "How do I reset my password?",  # L1
        "My API integration is returning 500 errors",  # L2
        "Our production database is down!",  # Escalation
    ]

    for query in queries:
        print(f"Query: {query}")
        print("-" * 50)

        result = await swarm.run(query)

        print(f"Response: {result.output}")
        print(f"Final agent: {result.final_agent}")
        print(f"Path: {' → '.join(result.execution_path)}")
        print()


if __name__ == "__main__":
    asyncio.run(main())
