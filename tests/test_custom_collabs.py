"""Unit tests for pydantic_collab.custom_collabs module.

Tests the specialized Collab classes: StarCollab, MeshCollab, PiplineCollab, HierarchyCollab.
"""

import pytest
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

from pydantic_collab import CollabAgent, CollabError
from pydantic_collab.custom_collabs import (
    HierarchyCollab,
    MeshCollab,
    PiplineCollab,
    StarCollab,
)


@pytest.fixture
def test_model():
    """Return a TestModel for deterministic testing."""
    return TestModel()


@pytest.fixture
def agent_a(test_model):
    return Agent(test_model, name='AgentA')


@pytest.fixture
def agent_b(test_model):
    return Agent(test_model, name='AgentB')


@pytest.fixture
def agent_c(test_model):
    return Agent(test_model, name='AgentC')


def make_collab_agent(agent, description):
    """Helper to create CollabAgent from Agent."""
    return CollabAgent(agent=agent, description=description)


class TestStarCollab:
    """Tests for StarCollab topology."""

    def test_basic_creation(self, agent_a, agent_b, agent_c):
        """Star collab should create with router calling others as tools."""
        agents = [
            (agent_a, 'Router agent'),
            (agent_b, 'Worker B'),
            (agent_c, 'Worker C'),
        ]
        collab = StarCollab(agents=agents)

        assert collab.starting_agent is not None
        assert collab.starting_agent.name == 'AgentA'
        assert collab.final_agent == collab.starting_agent

    def test_explicit_router_agent(self, agent_a, agent_b, agent_c):
        """Can specify which agent is the router."""
        agents = [
            (agent_a, 'Worker A'),
            (agent_b, 'Router agent'),
            (agent_c, 'Worker C'),
        ]
        collab = StarCollab(agents=agents, router_agent=(agent_b, 'Router agent'))

        assert collab.starting_agent.name == 'AgentB'
        assert collab.final_agent.name == 'AgentB'

    def test_tool_connections(self, agent_a, agent_b, agent_c):
        """Router should be able to call all other agents as tools."""
        agents = [
            (agent_a, 'Router'),
            (agent_b, 'Worker B'),
            (agent_c, 'Worker C'),
        ]
        collab = StarCollab(agents=agents)

        # Check tool connections
        tool_targets = collab.connections.get('AgentA', ())
        assert 'AgentB' in tool_targets
        assert 'AgentC' in tool_targets

    def test_no_handoffs(self, agent_a, agent_b):
        """Star topology should have no handoffs."""
        agents = [(agent_a, 'Router'), (agent_b, 'Worker')]
        collab = StarCollab(agents=agents)

        assert collab.max_handoffs == 0
        assert not collab.has_handoffs

    def test_final_agent_validation_in_star(self, agent_a, agent_b):
        """In StarCollab, final agent is always set to router (starting agent)."""
        agents = [(agent_a, 'Router'), (agent_b, 'Worker')]

        # StarCollab automatically sets final_agent = starting_agent
        collab = StarCollab(agents=agents, router_agent=(agent_a, 'Router'))

        # Both should be the same
        assert collab.final_agent == collab.starting_agent
        assert collab.final_agent.name == 'AgentA'

    def test_empty_agents_raises_error(self):
        """Creating StarCollab with no agents should raise error."""
        with pytest.raises(CollabError, match='No agents'):
            StarCollab(agents=[])


class TestMeshCollab:
    """Tests for MeshCollab topology."""

    def test_basic_creation(self, agent_a, agent_b, agent_c):
        """Mesh collab should allow all agents to call each other."""
        agents = [
            (agent_a, 'Agent A'),
            (agent_b, 'Agent B'),
            (agent_c, 'Agent C'),
        ]
        collab = MeshCollab(agents=agents)

        assert collab.starting_agent.name == 'AgentA'
        assert collab.final_agent == collab.starting_agent

    def test_all_agents_can_call_others(self, agent_a, agent_b, agent_c):
        """Each agent should be able to call all others."""
        agents = [
            (agent_a, 'Agent A'),
            (agent_b, 'Agent B'),
            (agent_c, 'Agent C'),
        ]
        collab = MeshCollab(agents=agents)

        # Starting agent can call B and C
        assert 'AgentB' in collab.connections.get('AgentA', ())
        assert 'AgentC' in collab.connections.get('AgentA', ())

        # Other agents can call everyone except themselves
        assert 'AgentA' in collab.connections.get('AgentB', ())
        assert 'AgentC' in collab.connections.get('AgentB', ())

        assert 'AgentA' in collab.connections.get('AgentC', ())
        assert 'AgentB' in collab.connections.get('AgentC', ())

    def test_no_handoffs(self, agent_a, agent_b):
        """Mesh topology should have no handoffs by default."""
        agents = [(agent_a, 'A'), (agent_b, 'B')]
        collab = MeshCollab(agents=agents)

        assert not collab.has_handoffs

    def test_agents_cannot_call_themselves(self, agent_a, agent_b):
        """Agent should not be in its own tool list."""
        agents = [(agent_a, 'A'), (agent_b, 'B')]
        collab = MeshCollab(agents=agents)

        assert 'AgentA' not in collab.connections.get('AgentA', ())
        assert 'AgentB' not in collab.connections.get('AgentB', ())


class TestPiplineCollab:
    """Tests for PiplineCollab (forward handoff chain)."""

    def test_basic_chain(self, agent_a, agent_b, agent_c):
        """Pipeline should create A→B→C handoff chain."""
        agents = [
            (agent_a, 'First'),
            (agent_b, 'Middle'),
            (agent_c, 'Last'),
        ]
        collab = PiplineCollab(agents=agents)

        assert collab.starting_agent.name == 'AgentA'
        assert collab.final_agent.name == 'AgentC'

    def test_handoff_connections(self, agent_a, agent_b, agent_c):
        """Each agent should hand off to the next in sequence."""
        agents = [
            (agent_a, 'First'),
            (agent_b, 'Middle'),
            (agent_c, 'Last'),
        ]
        collab = PiplineCollab(agents=agents)

        handoffs = collab.handoffs
        assert 'AgentB' in handoffs.get('AgentA', ())
        assert 'AgentC' in handoffs.get('AgentB', ())
        # Final agent has no handoffs
        assert handoffs.get('AgentC', ()) == ()

    def test_two_agent_pipeline(self, agent_a, agent_b):
        """Pipeline works with just two agents."""
        agents = [(agent_a, 'Start'), (agent_b, 'End')]
        collab = PiplineCollab(agents=agents)

        assert collab.starting_agent.name == 'AgentA'
        assert collab.final_agent.name == 'AgentB'
        assert 'AgentB' in collab.handoffs.get('AgentA', ())

    def test_single_agent_pipeline(self, agent_a):
        """Single agent pipeline - start is also final."""
        agents = [(agent_a, 'Only')]
        collab = PiplineCollab(agents=agents)

        assert collab.starting_agent.name == 'AgentA'
        assert collab.final_agent.name == 'AgentA'

    def test_final_agent_must_be_last(self, agent_a, agent_b, agent_c):
        """Final agent must be the last in the sequence."""
        agents = [
            (agent_a, 'First'),
            (agent_b, 'Middle'),
            (agent_c, 'Last'),
        ]

        collab_agent_b = CollabAgent(agent=agent_b, description='Middle')

        with pytest.raises(CollabError, match='Final agent must be last'):
            PiplineCollab(agents=agents, final_agent=collab_agent_b)

    def test_starting_agent_must_be_first(self, agent_a, agent_b, agent_c):
        """Starting agent must be first in the sequence."""
        agents = [
            (agent_a, 'First'),
            (agent_b, 'Middle'),
            (agent_c, 'Last'),
        ]

        collab_agent_b = CollabAgent(agent=agent_b, description='Middle')

        with pytest.raises(CollabError, match='starting agent must be first'):
            PiplineCollab(agents=agents, starting_agent=collab_agent_b)

    def test_has_handoffs(self, agent_a, agent_b):
        """Pipeline should have handoffs between consecutive agents."""
        agents = [(agent_a, 'Start'), (agent_b, 'End')]
        collab = PiplineCollab(agents=agents)

        assert collab.has_handoffs


class TestHierarchyCollab:
    """Tests for HierarchyCollab (planner + orchestrator pattern)."""

    def test_basic_creation(self, agent_a, agent_b, agent_c):
        """HierarchyCollab should initialize with planner and orchestrators."""
        planner = CollabAgent(agent=agent_a, description='Planner agent')
        orch_b = CollabAgent(agent=agent_b, description='Orchestrator B')
        orch_c = CollabAgent(agent=agent_c, description='Orchestrator C')

        orchestrator_agents = {
            orch_b: [],
            orch_c: [],
        }

        collab = HierarchyCollab(
            planner=planner, orchestrator_agents=orchestrator_agents
        )

        # Planner should be stored
        assert collab._planner.name == 'AgentA'

        # Orchestrator agents should be normalized
        assert len(collab._orchestrator_agents) == 2

    def test_orchestrator_with_tool_agents(self, agent_a, agent_b, agent_c, test_model):
        """Orchestrators can have tool agents assigned."""
        agent_d = Agent(test_model, name='AgentD')

        planner = CollabAgent(agent=agent_a, description='Planner')
        orch_b = CollabAgent(agent=agent_b, description='Orchestrator B')
        worker_c = CollabAgent(agent=agent_c, description='Worker C')
        worker_d = CollabAgent(agent=agent_d, description='Worker D')

        orchestrator_agents = {
            orch_b: [worker_c, worker_d],
        }

        collab = HierarchyCollab(
            planner=planner, orchestrator_agents=orchestrator_agents
        )

        # Check the orchestrator map was created
        assert len(collab._orchestrator_map) == 1

    def test_empty_orchestrators(self, agent_a):
        """HierarchyCollab can be created with empty orchestrator dict."""
        planner = CollabAgent(agent=agent_a, description='Planner')

        collab = HierarchyCollab(planner=planner, orchestrator_agents={})

        assert collab._planner.name == 'AgentA'
        assert len(collab._orchestrator_agents) == 0

    def test_topology_not_implemented(self, agent_a, agent_b):
        """HierarchyCollab._build_topology is currently a stub."""
        planner = CollabAgent(agent=agent_a, description='Planner')
        orch_b = CollabAgent(agent=agent_b, description='Orchestrator')

        orchestrator_agents = {orch_b: []}

        collab = HierarchyCollab(
            planner=planner, orchestrator_agents=orchestrator_agents
        )

        # The topology building is a pass statement
        # No handoffs or tool connections are set up
        assert collab.handoffs == {}


class TestCollabIntegration:
    """Integration tests for custom collabs."""

    @pytest.mark.asyncio
    async def test_star_collab_run(self, test_model):
        """StarCollab should run successfully with mocked model."""
        model = TestModel(custom_output_text='Final answer from router')

        router = Agent(model, name='Router')
        worker = Agent(model, name='Worker')

        collab = StarCollab(
            agents=[(router, 'Routes requests'), (worker, 'Does work')]
        )

        result = await collab.run('Test query')

        assert result.output == 'Final answer from router'
        assert result.final_agent == 'Router'

    @pytest.mark.asyncio
    async def test_pipeline_collab_run(self, test_model):
        """PiplineCollab should hand off through the chain."""
        from pydantic_collab._types import HandOffBase

        # Create models that produce handoffs
        model_a = TestModel(
            custom_output_args=HandOffBase(next_agent='AgentB', query='processed by A')
        )
        model_b = TestModel(custom_output_text='Final from B')

        agent_a = Agent(model_a, name='AgentA')
        agent_b = Agent(model_b, name='AgentB')

        collab = PiplineCollab(
            agents=[(agent_a, 'First stage'), (agent_b, 'Final stage')],
            max_handoffs=5,
        )

        result = await collab.run('Start query')

        assert 'AgentA' in result.execution_path
        assert 'AgentB' in result.execution_path
        assert result.final_agent == 'AgentB'

    @pytest.mark.asyncio
    async def test_mesh_collab_run(self, test_model):
        """MeshCollab should allow tool calls between agents."""
        model = TestModel(custom_output_text='Mesh result')

        agent_a = Agent(model, name='AgentA')
        agent_b = Agent(model, name='AgentB')

        collab = MeshCollab(
            agents=[(agent_a, 'Agent A'), (agent_b, 'Agent B')]
        )

        result = await collab.run('Test query')

        assert result.output == 'Mesh result'
        assert result.final_agent == 'AgentA'


class TestCollabWithCollabAgents:
    """Tests using CollabAgent instances directly."""

    def test_star_with_collab_agents(self, agent_a, agent_b):
        """StarCollab works with CollabAgent instances."""
        collab_a = CollabAgent(agent=agent_a, description='Router')
        collab_b = CollabAgent(agent=agent_b, description='Worker')

        collab = StarCollab(agents=[collab_a, collab_b])

        assert collab.starting_agent == collab_a
        assert 'AgentB' in collab.connections.get('AgentA', ())

    def test_pipeline_with_collab_agents(self, agent_a, agent_b, agent_c):
        """PiplineCollab works with CollabAgent instances."""
        collab_a = CollabAgent(agent=agent_a, description='First')
        collab_b = CollabAgent(agent=agent_b, description='Middle')
        collab_c = CollabAgent(agent=agent_c, description='Last')

        collab = PiplineCollab(agents=[collab_a, collab_b, collab_c])

        assert collab.starting_agent == collab_a
        assert collab.final_agent == collab_c
        assert 'AgentB' in collab.handoffs.get('AgentA', ())

    def test_mixed_agent_types(self, agent_a, agent_b, agent_c):
        """Can mix CollabAgent and tuple formats."""
        collab_a = CollabAgent(agent=agent_a, description='CollabAgent style')

        collab = PiplineCollab(
            agents=[collab_a, (agent_b, 'Tuple style'), (agent_c, 'Also tuple')]
        )

        assert len(collab._agents) == 3
        assert collab.starting_agent.name == 'AgentA'
