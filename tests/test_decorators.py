"""Unit tests for pydantic_collab decorator methods.

Tests the @tool, @tool_plain, and @toolset decorators on Collab instances.
"""

import pytest
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.test import TestModel
from pydantic_ai.toolsets import FunctionToolset

from pydantic_collab import Collab, CollabAgent
from pydantic_collab._types import HandOffBase
from pydantic_collab.custom_collabs import PipelineCollab, StarCollab


@pytest.fixture
def test_model():
    return TestModel()


@pytest.fixture
def simple_agents(test_model):
    """Create simple agents for testing decorators."""
    agent_a = Agent(test_model, name='AgentA')
    agent_b = Agent(test_model, name='AgentB')
    return agent_a, agent_b


class TestToolPlainDecorator:
    """Tests for @tool_plain decorator."""

    def test_tool_plain_basic(self, simple_agents):
        """Basic @tool_plain registers a tool for all agents."""
        agent_a, agent_b = simple_agents

        collab = StarCollab(agents=[(agent_a, 'A'), (agent_b, 'B')])

        @collab.tool_plain
        def add_numbers(x: int, y: int) -> int:
            """Add two numbers."""
            return x + y

        # Tool should be registered in _user_toolset
        assert len(collab._user_toolset.tools) == 1

    def test_tool_plain_with_name(self, simple_agents):
        """@tool_plain can specify a custom name."""
        agent_a, agent_b = simple_agents

        collab = StarCollab(agents=[(agent_a, 'A'), (agent_b, 'B')])

        @collab.tool_plain(name='custom_add')
        def add_numbers(x: int, y: int) -> int:
            """Add two numbers."""
            return x + y

        # Should have one tool with custom name
        assert len(collab._user_toolset.tools) == 1
        assert 'custom_add' in collab._user_toolset.tools

    def test_tool_plain_for_specific_agents(self, simple_agents):
        """@tool_plain can be limited to specific agents."""
        agent_a, agent_b = simple_agents

        collab = StarCollab(agents=[(agent_a, 'A'), (agent_b, 'B')])

        @collab.tool_plain(agents=['AgentA'])
        def only_for_a(x: int) -> int:
            """Only for agent A."""
            return x * 2

        # Tool registered in agent-specific toolset
        agent_a_collab = collab._name_to_agent['AgentA']
        agent_b_collab = collab._name_to_agent['AgentB']
        assert len(collab._toolset_by_agent[agent_a_collab].tools) == 1
        assert len(collab._toolset_by_agent[agent_b_collab].tools) == 0

    def test_tool_plain_multiple_agents(self, simple_agents):
        """@tool_plain can specify multiple agents."""
        agent_a, agent_b = simple_agents
        test_model = TestModel()
        agent_c = Agent(test_model, name='AgentC')

        collab = StarCollab(agents=[(agent_a, 'A'), (agent_b, 'B'), (agent_c, 'C')])

        @collab.tool_plain(agents=['AgentA', 'AgentB'])
        def for_a_and_b(x: int) -> int:
            """For agents A and B."""
            return x * 2

        agent_a_collab = collab._name_to_agent['AgentA']
        agent_b_collab = collab._name_to_agent['AgentB']
        agent_c_collab = collab._name_to_agent['AgentC']
        assert len(collab._toolset_by_agent[agent_a_collab].tools) == 1
        assert len(collab._toolset_by_agent[agent_b_collab].tools) == 1
        assert len(collab._toolset_by_agent[agent_c_collab].tools) == 0

    def test_multiple_tool_plain_decorators(self, simple_agents):
        """Can register multiple tools with @tool_plain."""
        agent_a, agent_b = simple_agents

        collab = StarCollab(agents=[(agent_a, 'A'), (agent_b, 'B')])

        @collab.tool_plain
        def tool_one(x: int) -> int:
            return x + 1

        @collab.tool_plain
        def tool_two(x: int) -> int:
            return x + 2

        assert len(collab._user_toolset.tools) == 2


class TestToolDecorator:
    """Tests for @tool decorator (with context)."""

    def test_tool_basic(self, simple_agents):
        """Basic @tool registers a tool with context access."""
        agent_a, agent_b = simple_agents

        collab = StarCollab(agents=[(agent_a, 'A'), (agent_b, 'B')])

        @collab.tool
        async def get_info(ctx: RunContext[None]) -> str:
            """Get info with context access."""
            return 'info'

        assert len(collab._user_toolset.tools) == 1

    def test_tool_with_custom_name(self, simple_agents):
        """@tool can specify a custom name."""
        agent_a, agent_b = simple_agents

        collab = StarCollab(agents=[(agent_a, 'A'), (agent_b, 'B')])

        @collab.tool(name='custom_info')
        async def get_info(ctx: RunContext[None]) -> str:
            """Get info."""
            return 'info'

        assert len(collab._user_toolset.tools) == 1
        assert 'custom_info' in collab._user_toolset.tools

    def test_tool_for_specific_agents(self, simple_agents):
        """@tool can be limited to specific agents."""
        agent_a, agent_b = simple_agents

        collab = StarCollab(agents=[(agent_a, 'A'), (agent_b, 'B')])

        @collab.tool(agents=['AgentB'])
        async def only_b_tool(ctx: RunContext[None]) -> str:
            """Only for B."""
            return 'b result'

        agent_a_collab = collab._name_to_agent['AgentA']
        agent_b_collab = collab._name_to_agent['AgentB']
        assert len(collab._toolset_by_agent[agent_b_collab].tools) == 1
        assert len(collab._toolset_by_agent[agent_a_collab].tools) == 0


class TestToolsetDecorator:
    """Tests for @toolset decorator."""

    def test_toolset_basic(self, simple_agents):
        """Basic @toolset registers a dynamic toolset."""
        agent_a, agent_b = simple_agents

        collab = StarCollab(agents=[(agent_a, 'A'), (agent_b, 'B')])

        @collab.toolset
        async def my_toolset(ctx: RunContext[None]):
            return FunctionToolset()

        # Toolset should be in dynamic_toolsets
        assert len(collab._dynamic_toolsets) == 1

    def test_toolset_with_per_run_step(self, simple_agents):
        """@toolset can specify per_run_step behavior."""
        agent_a, agent_b = simple_agents

        collab = StarCollab(agents=[(agent_a, 'A'), (agent_b, 'B')])

        @collab.toolset(per_run_step=False)
        async def my_toolset(ctx: RunContext[None]):
            return FunctionToolset()

        assert len(collab._dynamic_toolsets) == 1


class TestDecoratorIntegration:
    """Integration tests for decorators with actual runs."""

    @pytest.mark.asyncio
    async def test_tool_plain_execution(self):
        """Tool plain should be callable during run."""
        model = TestModel(custom_output_text='Used the tool')
        agent_a = Agent(model, name='AgentA')
        agent_b = Agent(model, name='AgentB')

        collab = StarCollab(agents=[(agent_a, 'A'), (agent_b, 'B')])

        call_count = 0

        @collab.tool_plain
        def increment() -> int:
            """Increment counter."""
            nonlocal call_count
            call_count += 1
            return call_count

        result = await collab.run('Call increment')

        # Model was run
        assert result.output == 'Used the tool'

    @pytest.mark.asyncio
    async def test_agent_specific_tool_in_pipeline(self):
        """Agent-specific tool should only be available to designated agents."""
        model_a = TestModel(
            custom_output_args=HandOffBase(next_agent='AgentB', query='continue')
        )
        model_b = TestModel(custom_output_text='Done')

        agent_a = Agent(model_a, name='AgentA')
        agent_b = Agent(model_b, name='AgentB')

        collab = PipelineCollab(
            agents=[(agent_a, 'First'), (agent_b, 'Second')], max_handoffs=5
        )

        tool_calls = []

        @collab.tool_plain(agents=['AgentB'])
        def b_only_tool(x: int) -> int:
            """Only B can use this."""
            tool_calls.append('b_tool')
            return x * 2

        result = await collab.run('Start')

        # Pipeline should complete
        assert result.final_agent == 'AgentB'


class TestDecoratorEdgeCases:
    """Edge cases and error handling for decorators."""

    def test_tool_with_collab_agent_reference(self, simple_agents):
        """Can reference agents by CollabAgent instance."""
        agent_a, agent_b = simple_agents
        collab_a = CollabAgent(agent=agent_a, description='A')
        collab_b = CollabAgent(agent=agent_b, description='B')

        collab = StarCollab(agents=[collab_a, collab_b])

        @collab.tool_plain(agents=[collab_a])
        def a_tool(x: int) -> int:
            return x

        agent_a_collab = collab._name_to_agent['AgentA']
        agent_b_collab = collab._name_to_agent['AgentB']
        assert len(collab._toolset_by_agent[agent_a_collab].tools) == 1
        assert len(collab._toolset_by_agent[agent_b_collab].tools) == 0

    def test_tool_plain_async_function(self, simple_agents):
        """@tool_plain works with async functions."""
        agent_a, agent_b = simple_agents

        collab = StarCollab(agents=[(agent_a, 'A'), (agent_b, 'B')])

        @collab.tool_plain
        async def async_tool(x: int) -> int:
            """Async tool."""
            return x * 2

        assert len(collab._user_toolset.tools) == 1

    def test_tool_returns_function(self, simple_agents):
        """@tool decorator returns the original function."""
        agent_a, agent_b = simple_agents

        collab = StarCollab(agents=[(agent_a, 'A'), (agent_b, 'B')])

        @collab.tool_plain
        def my_tool(x: int) -> int:
            return x

        # The decorated function should be callable
        assert my_tool(5) == 5

    def test_tool_preserves_docstring(self, simple_agents):
        """@tool preserves the function's docstring."""
        agent_a, agent_b = simple_agents

        collab = StarCollab(agents=[(agent_a, 'A'), (agent_b, 'B')])

        @collab.tool_plain
        def documented_tool(x: int) -> int:
            """This is the docstring."""
            return x

        assert 'docstring' in documented_tool.__doc__


class TestToolsParameter:
    """Tests for passing tools via constructor parameter."""

    def test_tools_in_constructor(self, simple_agents):
        """Can pass tools via constructor."""
        agent_a, agent_b = simple_agents

        def my_tool(x: int) -> int:
            """My tool."""
            return x * 2

        collab = StarCollab(
            agents=[(agent_a, 'A'), (agent_b, 'B')], tools=[my_tool]
        )

        assert len(collab._user_toolset.tools) == 1

    def test_single_tool_in_constructor(self, simple_agents):
        """Can pass single tool (not in list) via constructor."""
        agent_a, agent_b = simple_agents

        def my_tool(x: int) -> int:
            """My tool."""
            return x * 2

        collab = StarCollab(
            agents=[(agent_a, 'A'), (agent_b, 'B')], tools=[my_tool]
        )

        assert len(collab._user_toolset.tools) == 1

    def test_combine_constructor_and_decorator_tools(self, simple_agents):
        """Can combine constructor tools with decorator tools."""
        agent_a, agent_b = simple_agents

        def constructor_tool(x: int) -> int:
            """Constructor tool."""
            return x

        collab = StarCollab(
            agents=[(agent_a, 'A'), (agent_b, 'B')], tools=[constructor_tool]
        )

        @collab.tool_plain
        def decorator_tool(y: int) -> int:
            """Decorator tool."""
            return y

        assert len(collab._user_toolset.tools) == 2
