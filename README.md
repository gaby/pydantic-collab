# pydantic-collab

A Multi-Agent-System framework built on [pydantic-ai](https://ai.pydantic.dev/).

## Installation

```bash
pip install pydantic-collab
```

## Quick Start

Define agent topologies through custom or pre-build topologies. Agents communicate through *tool calls* (synchronous consultation) or *handoffs* (transfer of control).

```python
from pydantic_ai import Agent
from pydantic_ai.builtin_tools import WebSearchTool
from pydantic_collab import PipelineCollab

intake = Agent(name="Intake", system_prompt="Summarize requests and relevant data from the internet",
               builtin_tools=[WebSearchTool()])
reporter = Agent(name="Reporter", system_prompt="Create final response")

swarm = PipelineCollab(
    agents=[(intake, "Intake agent"), (reporter, "Reporter agent")],
    model="openai:gpt-5.2"
)

result = swarm.run_sync("Plan a birthday party for a celebrity that was born today")
print(result.output)
```

## Tool Calls and Handoffs

### Tool Calls (`agent_calls`)
Use when an agent needs help from another agent but is still in charge:
- Agent consults another agent synchronously
- Caller receives response and continues execution
- Memory can persist across calls (controllable by the caller agent)
- Depth of call recursion controlled by `max_agent_call_depth` (default: 3)
- Parallel execution enabled by default, can be enabled through `allow_parallel_agent_calls`

**Example:** Coordinator needs specialist input before deciding.

### Handoffs (`agent_handoffs`)
Use when an agent's part is done and control should transfer:
- Agent transfers control permanently
- Transferring agent stops processing
- Receiving agent gets context (configurable by the user and by the transferring agent)
- Counts toward `max_handoffs` limit

**Example:** Pipeline stages (Intake → Analysis → Report).

**Rule of thumb:** Tool calls for "help me with X", handoffs for "take over from here".

## Common Topologies

### Forward Chain Pipeline

```python
from pydantic_ai import Agent
from pydantic_collab import PipelineCollab

swarm = PipelineCollab(
    agents=[
        (Agent(name="Intake", system_prompt="Summarize and hand off"), "Intake"),
        (Agent(name="Analyst", system_prompt="Analyze and hand off"), "Analyst"),
        (Agent(name="Reporter", system_prompt="Create final response"), "Reporter"),
    ],
    model="openai:gpt-4o-mini",
)
```

### Star Topology

```python
from pydantic_ai import Agent
from pydantic_collab import StarCollab

swarm = StarCollab(
    agents=[
        (Agent(name="Coordinator", system_prompt="Route to specialists"), "Coordinator"),
        (Agent(name="L1Support", system_prompt="Handle simple issues"), "L1"),
        (Agent(name="L2Support", system_prompt="Handle complex issues"), "L2"),
    ],
    model="openai:gpt-4o-mini",
)
```

### Mesh Network

```python
from pydantic_ai import Agent
from pydantic_collab import MeshCollab

swarm = MeshCollab(
    agents=[
        (Agent(name="Strategist", system_prompt="Business strategy"), "Strategy"),
        (Agent(name="Technologist", system_prompt="Technical feasibility"), "Tech"),
        (Agent(name="Designer", system_prompt="User experience"), "Design"),
    ],
    model="openai:gemini-2.5-pro",
)
```

### Custom Topology

Define explicit tool calls and handoffs:

```python
from pydantic_collab import Collab, CollabAgent
from pydantic_ai import Agent

swarm = Collab(
    agents=[
        CollabAgent(
            agent=Agent(name="Router", system_prompt="Route requests"),
            description="Routes requests",
            agent_calls="Researcher",  # Can call as tool
            agent_handoffs="Writer",  # Can transfer control
        ),
        CollabAgent(
            agent=Agent(name="Researcher", system_prompt="Research topics"),
            description="Researches topics",
        ),
        CollabAgent(
            agent=Agent(name="Writer", system_prompt="Write content"),
            description="Writes content",
            agent_handoffs="Editor",
        ),
        CollabAgent(
            agent=Agent(name="Editor", system_prompt="Final editing"),
            description="Final editing",
        ),
    ],
    model="anthropic:claude-sonnet-4-5",
    final_agent="Editor",
)
```

## Visualizing Topology

Visualize your agent topology as a graph.
```python
swarm = Collab(...)
# Automatically opens image in a window (default behavior)
swarm.visualize_topology()
# Or save to file
swarm.visualize_topology(save_path="topology.png", show=False)
```

**Installation:** Requires visualization dependencies:
```bash
pip install pydantic-collab[viz]
```
### Example
![Topology_image](docs/topology_example.png)
## Adding Tools


### Custom Tools for All Agents

```python
swarm = Collab(agents=[...], model="openai:gpt-4o-mini")

@swarm.tool_plain
async def power(num1: int, num2: int) -> int:
    """Returns num1 powered by num2"""
    return num1 ** num2
```

### Custom Tools for Specific Agents

```python
@swarm.tool_plain(agents=("Researcher", "Analyst"))
async def fetch_data(url: str) -> str:
    """Fetch data from URL."""
    return f"Data from {url}"


@swarm.tool(agents=("Writer",))
async def save_draft(ctx: RunContext[MyDeps], content: str) -> str:
    """Save draft."""
    await ctx._deps.storage.save(content)
    return "Saved!"
```

## Result Object

```python
result = await swarm.run("Query")
# or
result = swarm.run_sync("Query")

result.output              # Final output
result.final_agent         # Agent that produced output
result.iterations          # Number of handoffs
result.execution_path      # ["Intake", "Analyst", "Reporter"]
result.execution_history   # Detailed step-by-step history
result.usage               # Token usage statistics
result.all_messages()     # Full message history

print(result.print_execution_flow())  # Visual flow diagram
```

## Configuration

### Execution Limits

```python
swarm = Collab(
    agents=[...],
    max_handoffs=10,              # Maximum handoff iterations (default: 10)
    max_agent_call_depth=3,       # Maximum recursive tool call depth (default: 3)
)
```

### Handoff Settings

Control what information flows between agents during handoffs.
Most settings accept three values: 
- *allow* - Agent decides every handoff
- *disallow" - Always false, agent has no say
- *allow* - Always true, agent has not say.

```python
from pydantic_collab import CollabSettings

swarm = Collab(
    agents=[...],
    swarm_settings=CollabSettings(
        include_conversation="allow",  # 
        include_thinking="disallow",  # Include thinking/reasoning parts
        include_handoff="allow",  # Accumulate previous handoff context
        include_tool_calls_with_callee="allow",  # Include tool calls with target agent
        output_restrictions="str_or_original",  # "only_str", "only_original", "str_or_original"
        include_topology_in_prompt=True,  # Show topology to agents (default: True)
    ),
)
```

### Custom Prompt Builder

```python
from pydantic_collab import Collab, PromptBuilderContext, CollabSettings


def my_prompt_builder(ctx: PromptBuilderContext) -> str:
    lines = [f"Agent: {ctx.agent.name}"]
    if ctx.can_handoff:
        lines.append(f"Hand off to: {', '.join(a.name for a in ctx.handoff_agents)}")
    return "\n".join(lines)


swarm = Collab(
    agents=[...],
    collab_settings=CollabSettings(prompt_builder=my_prompt_builder),
)
```

### Custom Context Builder

```python
from pydantic_collab import HandoffData, CollabSettings


def my_context_builder(data: HandoffData) -> str:
    parts = [f"From {data.caller_agent_name}:"]
    if data.message_history:
        parts.append("Previous conversation included")
    return "\n".join(parts)


swarm = Collab(
    agents=[...],
    swarm_settings=CollabSettings(context_builder=my_context_builder),
)
```

### Using Dependencies
```python
from pydantic import BaseModel
from pydantic_collab import Collab

class MyDeps(BaseModel):
    db: Database
    cache: Cache

swarm = Collab(
    agents=[...],
)
result = swarm.run_sync("...", deps=MyDeps(db=db, cache=cache))
```

## Examples

See `examples/` directory for complete working examples:

- [`01_simple_chain.py`](examples/01_simple_chain.py) - Basic forward handoff pipeline
- [`02_bidirectional_chain.py`](examples/02_bidirectional_chain.py) - Agents can handoff back
- [`04_mesh_network.py`](examples/04_mesh_network.py) - Full mesh collaboration
- [`08_mesh_with_tools.py`](examples/08_mesh_with_tools.py) - Mesh topology with function tools
- [`10_handoff_include_history.py`](examples/10_handoff_include_history.py) - Configuring handoff context
- [`12_data_analysis_pipeline.py`](examples/12_data_analysis_pipeline.py) - Complex multi-stage workflow
- [`create_topology_visualization.py`](examples/create_topology_visualization.py) - Generate topology visualization

Run examples:
```bash
uv run --env-file .env examples/01_simple_chain.py
```

## License

MIT
