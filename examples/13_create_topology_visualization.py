"""Create a sample topology visualization for the README.

Run with: uv run --env-file .env examples/13_create_topology_visualization.py
Requires: pip install pydantic-collab[viz] or uv add "pydantic-collab[viz]"
"""
from pydantic_collab import Collab, CollabAgent

# Create a content creation pipeline with reasonable agent names
swarm = Collab(
    agents=[
        CollabAgent(
            name="Researcher",
            system_prompt="Research topics",
            agent_calls=("Writer",),  # Can consult Writer
            agent_handoffs=("Writer", "Editor"),  # Can hand off to Writer or Editor
        ),
        CollabAgent(
            name="Writer",
            system_prompt="Write content",
            agent_calls=("Researcher", "Editor"),  # Can consult Researcher or Editor
            agent_handoffs=("Editor",),  # Can hand off to Editor
        ),
        CollabAgent(
            name="Editor",
            system_prompt="Edit and refine content",
            agent_calls=("Writer",),  # Can consult Writer
            agent_handoffs=("Publisher",),  # Can hand off to Publisher
        ),
        CollabAgent(
            name="Publisher",
            system_prompt="Finalize and publish",
        ),
    ],
    model="openai:gpt-4o-mini",
    starting_agent="Researcher",
    final_agent="Publisher",
    name="Content Creation Pipeline",
)

# Generate and save the visualization
swarm.visualize_topology(
    save_path="docs/topology_example.png",
    show=False,
    title="Content Creation Pipeline",
    figsize=(14, 10),
)

print("✓ Visualization saved to docs/topology_example.png")

