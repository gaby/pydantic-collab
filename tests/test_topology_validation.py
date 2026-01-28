"""Tests for handoff topology validation in Collab."""

import pytest
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

from pydantic_collab import Collab, CollabAgent, StarCollab
from pydantic_collab._types import CollabError


@pytest.fixture
def model() -> TestModel:
    """Provide a test model for all tests."""
    return TestModel()


class TestValidTopologies:
    """Test cases for valid handoff topologies that should pass validation."""

    def test_simple_chain(self, model: TestModel) -> None:
        """Valid chain: A -> B -> C (final)."""
        swarm = Collab(
            agents=[
                CollabAgent(agent=Agent(model, name='ChainA'), description='A', agent_handoffs=('ChainB',)),
                CollabAgent(agent=Agent(model, name='ChainB'), description='B', agent_handoffs=('ChainC',)),
                CollabAgent(agent=Agent(model, name='ChainC'), description='C'),
            ],
            final_agent='ChainC',
        )
        assert swarm.final_agent.name == 'ChainC'

    def test_cycle_with_exit(self, model: TestModel) -> None:
        """Valid cycle with exit: A <-> B, B -> C (final)."""
        swarm = Collab(
            agents=[
                CollabAgent(agent=Agent(model, name='CycleExitA'), description='A', agent_handoffs=('CycleExitB',)),
                CollabAgent(agent=Agent(model, name='CycleExitB'), description='B', agent_handoffs=('CycleExitA', 'CycleExitC')),
                CollabAgent(agent=Agent(model, name='CycleExitC'), description='C'),
            ],
            final_agent='CycleExitC',
        )
        assert swarm.final_agent.name == 'CycleExitC'

    def test_multiple_paths_to_final(self, model: TestModel) -> None:
        """Valid topology: A -> B -> D, A -> C -> D (final)."""
        swarm = Collab(
            agents=[
                CollabAgent(agent=Agent(model, name='MultiA'), description='A', agent_handoffs=('MultiB', 'MultiC')),
                CollabAgent(agent=Agent(model, name='MultiB'), description='B', agent_handoffs=('MultiD',)),
                CollabAgent(agent=Agent(model, name='MultiC'), description='C', agent_handoffs=('MultiD',)),
                CollabAgent(agent=Agent(model, name='MultiD'), description='D'),
            ],
            final_agent='MultiD',
        )
        assert swarm.final_agent.name == 'MultiD'

    def test_direct_to_final(self, model: TestModel) -> None:
        """Valid topology: A -> B (final) directly."""
        swarm = Collab(
            agents=[
                CollabAgent(agent=Agent(model, name='DirectA'), description='A', agent_handoffs=('DirectB',)),
                CollabAgent(agent=Agent(model, name='DirectB'), description='B'),
            ],
            final_agent='DirectB',
        )
        assert swarm.final_agent.name == 'DirectB'

    def test_start_is_final_no_validation_needed(self, model: TestModel) -> None:
        """When start == final, no handoff validation is needed."""
        swarm = StarCollab(
            agents=[
                CollabAgent(agent=Agent(model, name='StarCenter'), description='Center'),
                CollabAgent(agent=Agent(model, name='StarHelper'), description='Helper'),
            ],
        )
        # start == final, so validation is skipped
        assert swarm.starting_agent is swarm.final_agent


class TestDeadEndDetection:
    """Test cases for dead-end agent detection."""

    def test_dead_end_agent(self, model: TestModel) -> None:
        """Dead end: A -> B (no handoff), C is final but unreachable from B."""
        with pytest.raises(CollabError, match='has no handoff path to final agent'):
            Collab(
                agents=[
                    CollabAgent(agent=Agent(model, name='DeadA'), description='A', agent_handoffs=('DeadB',)),
                    CollabAgent(agent=Agent(model, name='DeadB'), description='B'),  # Dead end!
                    CollabAgent(agent=Agent(model, name='DeadC'), description='C'),
                ],
                final_agent='DeadC',
            )

    def test_branch_with_dead_end(self, model: TestModel) -> None:
        """One branch leads to final, another is dead end."""
        with pytest.raises(CollabError, match='Dead-end agents detected'):
            Collab(
                agents=[
                    CollabAgent(agent=Agent(model, name='BranchA'), description='A', agent_handoffs=('BranchB', 'BranchC')),
                    CollabAgent(agent=Agent(model, name='BranchB'), description='B', agent_handoffs=('BranchD',)),
                    CollabAgent(agent=Agent(model, name='BranchC'), description='C'),  # Dead end!
                    CollabAgent(agent=Agent(model, name='BranchD'), description='D'),
                ],
                final_agent='BranchD',
            )


class TestUnreachableFinalAgent:
    """Test cases for unreachable final agent detection."""

    def test_final_not_reachable(self, model: TestModel) -> None:
        """Final agent is not reachable from start."""
        with pytest.raises(CollabError, match='has no handoff path to final agent'):
            Collab(
                agents=[
                    CollabAgent(agent=Agent(model, name='UnreachA'), description='A', agent_handoffs=('UnreachB',)),
                    CollabAgent(agent=Agent(model, name='UnreachB'), description='B'),
                    CollabAgent(agent=Agent(model, name='UnreachC'), description='C'),  # Isolated final
                ],
                final_agent='UnreachC',
            )

    def test_disconnected_graph(self, model: TestModel) -> None:
        """Graph has disconnected components."""
        with pytest.raises(CollabError, match='has no handoff path to final agent'):
            Collab(
                agents=[
                    CollabAgent(agent=Agent(model, name='DisconnA'), description='A', agent_handoffs=('DisconnB',)),
                    CollabAgent(agent=Agent(model, name='DisconnB'), description='B'),
                    CollabAgent(agent=Agent(model, name='DisconnC'), description='C', agent_handoffs=('DisconnD',)),
                    CollabAgent(agent=Agent(model, name='DisconnD'), description='D'),  # Disconnected final
                ],
                final_agent='DisconnD',
            )


class TestInescapableCycles:
    """Test cases for inescapable cycle detection."""

    def test_simple_inescapable_cycle(self, model: TestModel) -> None:
        """Inescapable cycle: A <-> B, C is final but unreachable."""
        with pytest.raises(CollabError, match='has no handoff path to final agent'):
            Collab(
                agents=[
                    CollabAgent(agent=Agent(model, name='SimpleLoopA'), description='A', agent_handoffs=('SimpleLoopB',)),
                    CollabAgent(agent=Agent(model, name='SimpleLoopB'), description='B', agent_handoffs=('SimpleLoopA',)),
                    CollabAgent(agent=Agent(model, name='SimpleLoopC'), description='C'),
                ],
                final_agent='SimpleLoopC',
            )

    def test_cycle_on_branch_no_exit(self, model: TestModel) -> None:
        """Cycle on a branch with no exit: A -> B <-> C (cycle), A -> D (final).
        
        B and C form a cycle with no escape - this should fail because
        B and C are reachable but cannot reach final.
        """
        with pytest.raises(CollabError, match='Dead-end agents detected|Inescapable cycle'):
            Collab(
                agents=[
                    CollabAgent(agent=Agent(model, name='BranchLoopA'), description='A', agent_handoffs=('BranchLoopB', 'BranchLoopD')),
                    CollabAgent(agent=Agent(model, name='BranchLoopB'), description='B', agent_handoffs=('BranchLoopC',)),
                    CollabAgent(agent=Agent(model, name='BranchLoopC'), description='C', agent_handoffs=('BranchLoopB',)),
                    CollabAgent(agent=Agent(model, name='BranchLoopD'), description='D'),
                ],
                final_agent='BranchLoopD',
            )

    def test_cycle_with_exit_to_final(self, model: TestModel) -> None:
        """Cycle with exit: A -> B <-> C, B -> D (final) - should pass."""
        swarm = Collab(
            agents=[
                CollabAgent(agent=Agent(model, name='ExitLoopA'), description='A', agent_handoffs=('ExitLoopB',)),
                CollabAgent(agent=Agent(model, name='ExitLoopB'), description='B', agent_handoffs=('ExitLoopC', 'ExitLoopD')),
                CollabAgent(agent=Agent(model, name='ExitLoopC'), description='C', agent_handoffs=('ExitLoopB',)),
                CollabAgent(agent=Agent(model, name='ExitLoopD'), description='D'),
            ],
            final_agent='ExitLoopD',
        )
        assert swarm.final_agent.name == 'ExitLoopD'

    def test_three_agent_cycle_no_exit(self, model: TestModel) -> None:
        """Three-agent cycle with no exit: A -> B -> C -> A, D is final."""
        with pytest.raises(CollabError, match='has no handoff path to final agent|Inescapable cycle'):
            Collab(
                agents=[
                    CollabAgent(agent=Agent(model, name='TriLoopA'), description='A', agent_handoffs=('TriLoopB',)),
                    CollabAgent(agent=Agent(model, name='TriLoopB'), description='B', agent_handoffs=('TriLoopC',)),
                    CollabAgent(agent=Agent(model, name='TriLoopC'), description='C', agent_handoffs=('TriLoopA',)),
                    CollabAgent(agent=Agent(model, name='TriLoopD'), description='D'),
                ],
                final_agent='TriLoopD',
            )


class TestCollabProperties:
    """Tests for Collab properties and basic methods."""

    def test_agent_names_property(self, model: TestModel) -> None:
        """Test that agent_names returns all agent names."""
        collab = StarCollab(
            agents=[
                (Agent(model, name='AgentA'), 'A'),
                (Agent(model, name='AgentB'), 'B'),
            ]
        )
        names = collab.agent_names
        assert 'AgentA' in names
        assert 'AgentB' in names

    def test_repr(self, model: TestModel) -> None:
        """Test __repr__ method."""
        collab = Collab(
            agents=[
                CollabAgent(agent=Agent(model, name='StartAgent'), description='Start'),
            ],
            name='TestCollab',
        )
        r = repr(collab)
        assert 'Collab' in r
        assert 'StartAgent' in r

    def test_get_topology_ascii_no_handoffs(self, model: TestModel) -> None:
        """Test get_topology_ascii returns empty string when no handoffs."""
        collab = StarCollab(
            agents=[(Agent(model, name='AgentA'), 'A'), (Agent(model, name='AgentB'), 'B')]
        )
        assert collab.get_topology_ascii() == ''

    def test_get_topology_ascii_with_handoffs(self, model: TestModel) -> None:
        """Test get_topology_ascii with handoffs."""
        collab = Collab(
            agents=[
                CollabAgent(agent=Agent(model, name='AgentA'), description='A', agent_handoffs=('AgentB',)),
                CollabAgent(agent=Agent(model, name='AgentB'), description='B'),
            ],
            final_agent='AgentB',
        )
        ascii_top = collab.get_topology_ascii()
        assert 'AgentA' in ascii_top
        assert 'AgentB' in ascii_top
        assert 'Handoffs' in ascii_top


class TestCollabNormalization:
    """Tests for agent normalization methods."""

    def test_normalize_agent_string_not_found(self, model: TestModel) -> None:
        """Test _normalize_agent raises error for unknown string name."""
        collab = StarCollab(
            agents=[(Agent(model, name='AgentA'), 'A')]
        )
        with pytest.raises(CollabError, match="not found"):
            collab._normalize_agent('NonExistent')

    def test_normalize_agent_string_before_init(self, model: TestModel) -> None:
        """Test _normalize_agent with string before collab is initialized."""
        collab = StarCollab(
            agents=[(Agent(model, name='AgentA'), 'A')]
        )
        # Temporarily clear name_to_agent
        collab._name_to_agent = {}
        with pytest.raises(CollabError, match="not yet initialized"):
            collab._normalize_agent('SomeAgent')

    def test_normalize_agent_tuple_requires_description(self, model: TestModel) -> None:
        """Test _normalize_agent raises error when description required but missing."""
        collab = StarCollab(
            agents=[(Agent(model, name='AgentA'), 'A')]
        )
        agent = Agent(model, name='NewAgent')
        with pytest.raises(CollabError, match="Description required"):
            collab._normalize_agent((agent, None), require_description=True)

    def test_normalize_agent_abstract_requires_description(self, model: TestModel) -> None:
        """Test _normalize_agent raises error for AbstractAgent without description."""
        collab = StarCollab(
            agents=[(Agent(model, name='AgentA'), 'A')]
        )
        agent = Agent(model, name='NewAgent')
        with pytest.raises(CollabError, match="Description required"):
            collab._normalize_agent(agent, require_description=True)

    def test_normalize_and_reuse_existing_agent(self, model: TestModel) -> None:
        """Test _normalize_and_reuse_agent returns existing agent."""
        agent = Agent(model, name='AgentA')
        collab = StarCollab(
            agents=[(agent, 'A')]
        )
        # Pass the same underlying agent
        result = collab._normalize_and_reuse_agent(agent)
        assert result is collab._agents[0]


class TestCollabInstructions:
    """Tests for instructions handling."""

    def test_string_instructions(self, model: TestModel) -> None:
        """Test that string instructions are wrapped in list."""
        collab = Collab(
            agents=[CollabAgent(agent=Agent(model, name='AgentA'), description='A')],
            instructions='Test instruction',
        )
        assert collab._instructions == ['Test instruction']

    def test_list_instructions(self, model: TestModel) -> None:
        """Test that list instructions are preserved."""
        instructions = ['Inst 1', 'Inst 2']
        collab = Collab(
            agents=[CollabAgent(agent=Agent(model, name='AgentA'), description='A')],
            instructions=instructions,
        )
        assert collab._instructions == instructions

    def test_none_instructions(self, model: TestModel) -> None:
        """Test that None instructions become empty list."""
        collab = Collab(
            agents=[CollabAgent(agent=Agent(model, name='AgentA'), description='A')],
            instructions=None,
        )
        assert collab._instructions == []


class TestCollabBuildTopology:
    """Tests for _build_topology method."""

    def test_starting_agent_not_in_agents_gets_added(self, model: TestModel) -> None:
        """Test starting agent not in agents list gets added."""
        agent_a = Agent(model, name='AgentA')
        agent_b = Agent(model, name='AgentB')

        collab_a = CollabAgent(agent=agent_a, description='A')
        collab_b = CollabAgent(agent=agent_b, description='B')

        collab = Collab(
            agents=[collab_b],
            starting_agent=collab_a,
        )
        assert collab.starting_agent is collab_a
        assert collab_a.name in collab.agent_names

    def test_final_agent_not_in_agents_gets_added(self, model: TestModel) -> None:
        """Test final agent not in agents list gets added."""
        agent_a = Agent(model, name='AddFinalA')
        agent_b = Agent(model, name='AddFinalB')

        collab_a = CollabAgent(agent=agent_a, description='A', agent_handoffs=['AddFinalB'])
        collab_b = CollabAgent(agent=agent_b, description='B')

        collab = Collab(
            agents=[collab_a],
            starting_agent=collab_a,
            final_agent=collab_b,
        )
        assert collab.final_agent is collab_b
        assert collab_b.name in collab.agent_names

    def test_agent_cannot_call_itself(self, model: TestModel) -> None:
        """Test that agent cannot have itself in agent_calls."""
        with pytest.raises(CollabError, match="cannot call itself"):
            Collab(
                agents=[
                    CollabAgent(agent=Agent(model, name='SelfCall'), description='Self', agent_calls=['SelfCall']),
                ],
            )

    def test_agent_cannot_handoff_to_itself(self, model: TestModel) -> None:
        """Test that agent cannot handoff to itself."""
        with pytest.raises(CollabError, match="cannot call itself"):
            Collab(
                agents=[
                    CollabAgent(agent=Agent(model, name='SelfHandoff'), description='Self', agent_handoffs=['SelfHandoff']),
                    CollabAgent(agent=Agent(model, name='Final'), description='Final'),
                ],
                final_agent='Final',
            )


class TestCollabAgentDiscovery:
    """Tests for automatic agent discovery from calls/handoffs."""

    def test_agents_discovered_from_agent_calls(self, model: TestModel) -> None:
        """Test agents referenced in agent_calls are automatically added."""
        hidden_agent = CollabAgent(agent=Agent(model, name='HiddenAgent'), description='Hidden')

        collab = Collab(
            agents=[
                CollabAgent(
                    agent=Agent(model, name='MainAgent'),
                    description='Main',
                    agent_calls=[hidden_agent],  # Reference by CollabAgent object
                ),
            ],
        )
        # HiddenAgent should be discovered and added
        assert 'HiddenAgent' in collab.agent_names

    def test_agents_discovered_from_agent_handoffs(self, model: TestModel) -> None:
        """Test agents referenced in agent_handoffs are automatically added."""
        hidden_agent = CollabAgent(agent=Agent(model, name='HiddenHandoff'), description='Hidden')

        collab = Collab(
            agents=[
                CollabAgent(
                    agent=Agent(model, name='MainAgent'),
                    description='Main',
                    agent_handoffs=[hidden_agent],
                ),
            ],
            final_agent=hidden_agent,
        )
        assert 'HiddenHandoff' in collab.agent_names


class TestGetNameToAgent:
    """Tests for _get_name_to_agent method."""

    def test_get_name_to_agent_success(self, model: TestModel) -> None:
        """Test successful agent lookup by name."""
        collab = StarCollab(agents=[(Agent(model, name='AgentA'), 'A')])
        agent = collab._get_name_to_agent('AgentA')
        assert agent.name == 'AgentA'

    def test_get_name_to_agent_not_found_raises(self, model: TestModel) -> None:
        """Test agent not found raises CollabError."""
        collab = StarCollab(agents=[(Agent(model, name='AgentA'), 'A')])
        with pytest.raises(CollabError, match='not found'):
            collab._get_name_to_agent('NonExistent')

    def test_get_name_to_agent_custom_error(self, model: TestModel) -> None:
        """Test agent not found with custom error message."""
        collab = StarCollab(agents=[(Agent(model, name='AgentA'), 'A')])
        with pytest.raises(CollabError, match='Custom error'):
            collab._get_name_to_agent('NonExistent', exception='Custom error')

    def test_get_name_to_agent_no_exception(self, model: TestModel) -> None:
        """Test agent not found returns None when exception=False."""
        collab = StarCollab(agents=[(Agent(model, name='AgentA'), 'A')])
        result = collab._get_name_to_agent('NonExistent', exception=False)
        assert result is None


class TestOutputTypeForAgent:
    """Tests for get_output_type_for_agent method."""

    def test_output_type_only_str(self, model: TestModel) -> None:
        """Test output_restrictions='only_str' results in str type."""
        from pydantic_collab._types import CollabSettings

        collab = Collab(
            agents=[CollabAgent(agent=Agent(model, name='AgentA'), description='A')],
            collab_settings=CollabSettings(output_restrictions='only_str'),
        )
        output_type = collab.get_output_type_for_agent(collab._agents[0], None)
        assert output_type is str

    def test_output_type_only_original(self, model: TestModel) -> None:
        """Test output_restrictions='only_original' uses agent's output type."""
        from pydantic_collab._types import CollabSettings

        collab = Collab(
            agents=[CollabAgent(agent=Agent(model, name='AgentA', output_type=int), description='A')],
            collab_settings=CollabSettings(output_restrictions='only_original'),
        )
        output_type = collab.get_output_type_for_agent(collab._agents[0], None)
        assert output_type is int


class TestValidateAgents:
    """Tests for agent validation."""

    def test_duplicate_agent_names_raises(self, model: TestModel) -> None:
        """Test that duplicate agent names raise an error."""
        agent1 = Agent(model, name='DupeName')
        agent2 = Agent(model, name='DupeName')

        with pytest.raises(ValueError, match='same Agent name'):
            Collab(
                agents=[
                    CollabAgent(agent=agent1, description='First'),
                    CollabAgent(agent=agent2, description='Second'),
                ],
            )

    def test_non_starting_agent_without_description_not_raises(self, model: TestModel) -> None:
        """Test non-starting agent without description raises error."""
        agent_a = Agent(model, name='AgentA')
        agent_b = Agent(model, name='AgentB')

        # AgentB will be added via agent_calls but has no description
        assert Collab(
            agents=[
                CollabAgent(agent=agent_a, description='A', agent_calls=['AgentB']),
                CollabAgent(agent=agent_b, description=None),  # No description, not starting
            ],
        )


class TestAllowBackHandoff:
    """Tests for allow_back_handoff configuration."""

    def test_disallow_back_handoff(self, model: TestModel) -> None:
        """Test that disallowed back handoffs are filtered out."""
        collab = Collab(
            agents=[
                CollabAgent(agent=Agent(model, name='AgentA'), description='A', agent_handoffs=['AgentB']),
                CollabAgent(agent=Agent(model, name='AgentB'), description='B', agent_handoffs=['AgentA', 'AgentC']),
                CollabAgent(agent=Agent(model, name='AgentC'), description='C'),
            ],
            final_agent='AgentC',
            allow_back_handoff=False,
        )
        # Simulate that AgentA has already handed off
        collab._already_handed_off.add(collab._name_to_agent['AgentA'])

        # When checking allowed targets for AgentB, AgentA should be excluded
        agent_b = collab._name_to_agent['AgentB']
        targets = collab._get_allowed_handoff_targets(agent_b)
        target_names = [t.name for t in targets]
        assert 'AgentA' not in target_names
        assert 'AgentC' in target_names


class TestRunSync:
    """Tests for run_sync method."""

    def test_run_sync_executes(self, model: TestModel) -> None:
        """Test run_sync executes and returns result."""
        from pydantic_ai.models.test import TestModel as TM

        test_model = TM(custom_output_text='Sync result')
        collab = StarCollab(
            agents=[
                (Agent(test_model, name='Router'), 'Router'),
                (Agent(test_model, name='Worker'), 'Worker'),
            ]
        )
        result = collab.run_sync('Test query')
        assert result.output == 'Sync result'


class TestGetTsByAgents:
    """Tests for _get_ts_by_agents method."""

    def test_get_ts_by_agents_none(self, model: TestModel) -> None:
        """Test _get_ts_by_agents with None returns user toolset."""
        collab = StarCollab(agents=[(Agent(model, name='AgentA'), 'A')])
        result = collab._get_ts_by_agents(None)
        assert len(result) == 1
        assert result[0] is collab._user_toolset

    def test_get_ts_by_agents_string(self, model: TestModel) -> None:
        """Test _get_ts_by_agents with string agent name."""
        collab = StarCollab(agents=[(Agent(model, name='AgentA'), 'A')])
        result = collab._get_ts_by_agents('AgentA')
        assert len(result) == 1

    def test_get_ts_by_agents_collab_agent(self, model: TestModel) -> None:
        """Test _get_ts_by_agents with CollabAgent."""
        collab_a = CollabAgent(agent=Agent(model, name='AgentA'), description='A')
        collab = StarCollab(agents=[collab_a])
        result = collab._get_ts_by_agents(collab_a)
        assert len(result) == 1

    def test_get_ts_by_agents_abstract_agent(self, model: TestModel) -> None:
        """Test _get_ts_by_agents with AbstractAgent."""
        agent = Agent(model, name='AgentA')
        collab = StarCollab(agents=[(agent, 'A')])
        result = collab._get_ts_by_agents(agent)
        assert len(result) == 1


class TestVisualization:
    """Tests for visualization methods."""

    def test_visualize_topology_calls_render(self, model: TestModel) -> None:
        """Test visualize_topology calls render_topology."""
        pytest.importorskip("matplotlib")
        pytest.importorskip("networkx")
        import matplotlib.pyplot as plt

        collab = StarCollab(
            agents=[(Agent(model, name='AgentA'), 'A'), (Agent(model, name='AgentB'), 'B')]
        )
        fig = collab.visualize_topology(show=False)
        assert fig is not None
        plt.close(fig)


class TestRequireStartAgentIsFinalAgent:
    """Tests for _require_start_agent_is_final_agent method."""

    def test_raises_when_different(self, model: TestModel) -> None:
        """Test raises error when starting != final."""
        collab = Collab(
            agents=[
                CollabAgent(agent=Agent(model, name='StartNotFinalA'), description='A', agent_handoffs=['StartNotFinalB']),
                CollabAgent(agent=Agent(model, name='StartNotFinalB'), description='B'),
            ],
            final_agent='StartNotFinalB',
        )
        with pytest.raises(CollabError, match='starting agent'):
            collab._require_start_agent_is_final_agent()

    def test_raises_custom_message(self, model: TestModel) -> None:
        """Test raises with custom error message."""
        collab = Collab(
            agents=[
                CollabAgent(agent=Agent(model, name='CustomMsgA'), description='A', agent_handoffs=['CustomMsgB']),
                CollabAgent(agent=Agent(model, name='CustomMsgB'), description='B'),
            ],
            final_agent='CustomMsgB',
        )
        with pytest.raises(CollabError, match='Custom message'):
            collab._require_start_agent_is_final_agent('Custom message')


class TestEnsureAgentPresent:
    """Tests for _ensure_agent_present method."""

    def test_returns_agent_when_present(self, model: TestModel) -> None:
        """Test returns agent when not None."""
        collab = StarCollab(agents=[(Agent(model, name='EnsureA'), 'A')])
        agent = collab._agents[0]
        result = collab._ensure_agent_present(agent)
        assert result is agent

    def test_raises_when_none(self, model: TestModel) -> None:
        """Test raises when agent is None."""
        collab = StarCollab(agents=[(Agent(model, name='EnsureB'), 'A')])
        with pytest.raises(CollabError, match='Agent must be present'):
            collab._ensure_agent_present(None)

    def test_raises_custom_message(self, model: TestModel) -> None:
        """Test raises with custom message."""
        collab = StarCollab(agents=[(Agent(model, name='EnsureC'), 'A')])
        with pytest.raises(CollabError, match='Custom error here'):
            collab._ensure_agent_present(None, 'Custom error here')


class TestGetNameFromAgentDesc:
    """Tests for _get_name_from_agentdesc method."""

    def test_with_collab_agent(self, model: TestModel) -> None:
        """Test with CollabAgent returns name."""
        collab = StarCollab(agents=[(Agent(model, name='DescA'), 'A')])
        name = collab._get_name_from_agentdesc(collab._agents[0])
        assert name == 'DescA'

    def test_with_abstract_agent(self, model: TestModel) -> None:
        """Test with AbstractAgent returns name."""
        agent = Agent(model, name='DescB')
        collab = StarCollab(agents=[(agent, 'A')])
        name = collab._get_name_from_agentdesc(agent)
        assert name == 'DescB'

    def test_with_string(self, model: TestModel) -> None:
        """Test with string returns string."""
        collab = StarCollab(agents=[(Agent(model, name='DescC'), 'A')])
        name = collab._get_name_from_agentdesc('SomeAgentName')
        assert name == 'SomeAgentName'


class TestGetExplicitHandoffModel:
    """Tests for _get_explicit_handoff_model method."""

    def test_empty_handoff_agents_raises(self, model: TestModel) -> None:
        """Test that empty handoff agents raises ValueError."""
        collab = StarCollab(agents=[(Agent(model, name='HandoffModelA'), 'A')])
        with pytest.raises(ValueError, match='No handoff agents'):
            collab._get_explicit_handoff_model([])


class TestLogfireInit:
    """Tests for _init_logfire method."""

    def test_logfire_not_installed(self, model: TestModel) -> None:
        """Test _init_logfire when logfire is not installed."""
        import sys
        from unittest.mock import patch

        # Mock logfire import to raise ImportError
        with patch.dict(sys.modules, {'logfire': None}):
            collab = Collab(
                agents=[CollabAgent(agent=Agent(model, name='LogfireTestA'), description='A')],
                instrument_logfire=True,
            )
            assert collab._logfire is None

    def test_logfire_disabled(self, model: TestModel) -> None:
        """Test _init_logfire when instrument_logfire=False."""
        collab = Collab(
            agents=[CollabAgent(agent=Agent(model, name='LogfireTestB'), description='A')],
            instrument_logfire=False,
        )
        assert collab._logfire is None


class TestGetTsByAgentsCollection:
    """Tests for _get_ts_by_agents with collections."""

    def test_get_ts_by_agents_list(self, model: TestModel) -> None:
        """Test _get_ts_by_agents with list of agents."""
        agent_a = Agent(model, name='TsListA')
        agent_b = Agent(model, name='TsListB')
        collab = StarCollab(agents=[(agent_a, 'A'), (agent_b, 'B')])
        result = collab._get_ts_by_agents(['TsListA', 'TsListB'])
        assert len(result) == 2


class TestRunWithDeps:
    """Tests for running Collab with dependencies."""

    @pytest.mark.asyncio
    async def test_run_with_dict_deps(self, model: TestModel) -> None:
        """Test run with dictionary of deps per agent."""
        from pydantic_ai.models.test import TestModel as TM

        test_model = TM(custom_output_text='Result with deps')

        agent_a = Agent(test_model, name='DepsAgentA', deps_type=str)
        collab = StarCollab(
            agents=[(agent_a, 'A')],
        )

        # Run with dict deps
        result = await collab.run('Test query', deps={'DepsAgentA': 'my_dep_value'})
        assert result.output == 'Result with deps'

    @pytest.mark.asyncio
    async def test_run_with_single_deps(self, model: TestModel) -> None:
        """Test run with single deps value for all agents."""
        from pydantic_ai.models.test import TestModel as TM

        test_model = TM(custom_output_text='Single deps result')

        agent_a = Agent(test_model, name='SingleDepsA', deps_type=str)
        collab = StarCollab(
            agents=[(agent_a, 'A')],
        )

        # Run with single deps value
        result = await collab.run('Test query', deps='shared_dep')
        assert result.output == 'Single deps result'


class TestTopologyAsciiNoHandoffPairs:
    """Tests for get_topology_ascii edge cases."""

    def test_ascii_with_agent_but_no_handoff_pairs(self, model: TestModel) -> None:
        """Test get_topology_ascii when there are handoffs dict but no actual pairs."""
        collab = Collab(
            agents=[
                CollabAgent(agent=Agent(model, name='AsciiNoHOA'), description='A', agent_handoffs=['AsciiNoHOB']),
                CollabAgent(agent=Agent(model, name='AsciiNoHOB'), description='B'),
            ],
            final_agent='AsciiNoHOB',
        )

        # The ASCII should include handoff info
        ascii_output = collab.get_topology_ascii()
        assert 'Handoffs' in ascii_output
        assert 'AsciiNoHOA→AsciiNoHOB' in ascii_output


class TestLogfireConfigured:
    """Tests for logfire initialization when configured."""

    def test_logfire_configured_but_not_initialized(self, model: TestModel) -> None:
        """Test _init_logfire when logfire is installed but not configured."""
        import sys
        from unittest.mock import MagicMock, patch

        # Create a mock logfire module that's "installed" but not configured
        mock_logfire = MagicMock()
        mock_logfire.DEFAULT_LOGFIRE_INSTANCE.config._initialized = False

        with patch.dict(sys.modules, {'logfire': mock_logfire}):
            collab = Collab(
                agents=[CollabAgent(agent=Agent(model, name='LogfireConfigA'), description='A')],
                instrument_logfire=True,
            )
            # Should fall back to standard logging since logfire isn't configured
            assert collab._logfire is None

    def test_logfire_attribute_error(self, model: TestModel) -> None:
        """Test _init_logfire when logfire has no _initialized attribute."""
        import sys
        from unittest.mock import MagicMock, patch

        # Create a mock config that raises AttributeError on _initialized access
        mock_config = MagicMock()
        del mock_config._initialized  # Remove auto-created attribute
        mock_config.configure_mock(**{'_initialized': MagicMock(side_effect=AttributeError)})

        mock_logfire = MagicMock()
        # Make accessing _initialized raise AttributeError
        type(mock_logfire.DEFAULT_LOGFIRE_INSTANCE.config).__getattr__ = lambda self, name: (_ for _ in ()).throw(AttributeError(name)) if name == '_initialized' else MagicMock()

        with patch.dict(sys.modules, {'logfire': mock_logfire}):
            collab = Collab(
                agents=[CollabAgent(agent=Agent(model, name='LogfireAttrA'), description='A')],
                instrument_logfire=True,
            )
            # Should handle AttributeError gracefully (configured=False, so logfire=None)
            assert collab._logfire is None

    def test_logfire_configured_and_initialized(self, model: TestModel) -> None:
        """Test _init_logfire when logfire is properly configured."""
        import sys
        from unittest.mock import MagicMock, patch

        # Create a mock logfire module that's fully configured
        mock_logfire = MagicMock()
        mock_logfire.DEFAULT_LOGFIRE_INSTANCE.config._initialized = True

        with patch.dict(sys.modules, {'logfire': mock_logfire}):
            collab = Collab(
                agents=[CollabAgent(agent=Agent(model, name='LogfireInitA'), description='A')],
                instrument_logfire=True,
            )
            # Should use logfire as the logger
            assert collab._logfire is mock_logfire
            assert collab._logger is mock_logfire


class TestRunAgentValidation:
    """Tests for _run_agent method validation."""

    @pytest.mark.asyncio
    async def test_run_agent_both_history_and_context_raises(self, model: TestModel) -> None:
        """Test that providing both message_history and context_from_handoff raises error."""
        from pydantic_ai.models.test import TestModel as TM
        from pydantic_ai import RunUsage
        from pydantic_collab._types import CollabState

        test_model = TM(custom_output_text='Result')
        agent = Agent(test_model, name='ValidateAgent')
        collab = StarCollab(agents=[(agent, 'A')])

        state = CollabState(query='test')
        usage = RunUsage()

        with pytest.raises(CollabError, match="Can't have both message_history and context_from_handoff"):
            await collab._run_agent(
                collab._agents[0],
                'test',
                context_from_handoff='some context',
                message_history=[{'role': 'user', 'content': 'test'}],
                state=state,
                usage=usage,
            )

    @pytest.mark.asyncio
    async def test_run_agent_without_deps_defaults_to_empty_dict(self, model: TestModel) -> None:
        """Test that _run_agent works when deps is not provided (defaults to {})."""
        from pydantic_ai.models.test import TestModel as TM
        from pydantic_ai import RunUsage
        from pydantic_collab._types import CollabState

        test_model = TM(custom_output_text='Result without deps')
        agent = Agent(test_model, name='NoDepsAgent')
        collab = StarCollab(agents=[(agent, 'A')])

        state = CollabState(query='test')
        usage = RunUsage()

        # Call _run_agent without passing deps - it should default to None internally
        # which then gets converted to {} at line 725
        result = await collab._run_agent(
            collab._agents[0],
            'test query',
            context_from_handoff=None,
            message_history=None,
            state=state,
            usage=usage,
            # deps is not passed, so it defaults to None
        )

        assert result.output == 'Result without deps'


class TestIsInstrumented:
    """Tests for is_instrumented property."""

    def test_is_instrumented_false_by_default(self, model: TestModel) -> None:
        """Test is_instrumented is False when no agents have _instrument_default."""
        collab = StarCollab(agents=[(Agent(model, name='NotInstrA'), 'A')])
        assert collab.is_instrumented is False

    def test_is_instrumented_true_when_collab_agent_instrumented(self, model: TestModel) -> None:
        """Test is_instrumented is True when a CollabAgent has _instrument_default=True."""
        agent = Agent(model, name='InstrumentedA')
        collab = StarCollab(agents=[(agent, 'A')])

        # The code checks CollabAgent objects in self._agents
        # Set the attribute on the CollabAgent wrapper
        object.__setattr__(collab._agents[0], '_instrument_default', True)

        assert collab.is_instrumented is True


class TestRunWithLogfire:
    """Tests for run method with logfire integration."""

    @pytest.mark.asyncio
    async def test_run_with_logfire_span(self, model: TestModel) -> None:
        """Test run method creates logfire span when logfire is configured."""
        import sys
        from unittest.mock import MagicMock, patch, AsyncMock

        from pydantic_ai.models.test import TestModel as TM

        test_model = TM(custom_output_text='Logfire result')
        agent = Agent(test_model, name='LogfireRunA')

        # Create mock logfire
        mock_span = MagicMock()
        mock_span.__enter__ = MagicMock(return_value=mock_span)
        mock_span.__exit__ = MagicMock(return_value=None)

        mock_logfire = MagicMock()
        mock_logfire.DEFAULT_LOGFIRE_INSTANCE.config._initialized = True
        mock_logfire.span.return_value = mock_span

        with patch.dict(sys.modules, {'logfire': mock_logfire}):
            collab = StarCollab(
                agents=[(agent, 'A')],
                instrument_logfire=True,
            )
            result = await collab.run('Test query')

            # Verify span was used
            assert mock_logfire.span.called
            assert mock_span.__enter__.called
            assert mock_span.__exit__.called


class TestRunExceptionHandling:
    """Tests for exception handling in run method."""

    @pytest.mark.asyncio
    async def test_run_handles_agent_run_error(self, model: TestModel) -> None:
        """Test run method re-raises AgentRunError."""
        from unittest.mock import patch, AsyncMock
        from pydantic_ai.exceptions import AgentRunError
        from pydantic_ai.models.test import TestModel as TM

        test_model = TM(custom_output_text='Result')
        agent = Agent(test_model, name='ErrorAgent')
        collab = StarCollab(agents=[(agent, 'A')])

        # Mock _run_agent to raise AgentRunError
        with patch.object(collab, '_run_agent', new_callable=AsyncMock) as mock_run:
            mock_run.side_effect = AgentRunError('Test error')

            with pytest.raises(AgentRunError, match='Test error'):
                await collab.run('Test query')


class TestAgentToolCallExceptionHandling:
    """Tests for exception handling in agent tool calls."""

    @pytest.mark.asyncio
    async def test_agent_tool_call_handles_exception(self, model: TestModel) -> None:
        """Test that agent tool call exception is wrapped in ModelRetry."""
        from pydantic_ai.models.test import TestModel as TM
        from pydantic_ai import RunUsage
        from pydantic_collab._types import CollabState
        from pydantic_ai import ModelRetry
        from unittest.mock import AsyncMock, MagicMock, patch

        test_model = TM(custom_output_text='Result')

        # Create caller agent that can call helper
        caller = Agent(test_model, name='CallerAgent')
        helper = Agent(test_model, name='HelperAgent')

        from pydantic_collab import CollabAgent
        collab = Collab(
            agents=[
                CollabAgent(agent=caller, description='Caller', agent_calls=['HelperAgent']),
                CollabAgent(agent=helper, description='Helper'),
            ]
        )

        state = CollabState(query='test')
        deps: dict = {}

        # Get the tool that allows calling other agents
        tool = collab._get_agent_call_tool(
            collab._agents[0],  # Caller
            state,
            max_agent_calls_depths_allowed=3,
            deps=deps,
        )

        # Create a mock context
        mock_ctx = MagicMock()
        mock_ctx.usage = RunUsage()

        # Mock _run_agent to raise an exception
        with patch.object(collab, '_run_agent', new_callable=AsyncMock) as mock_run:
            mock_run.side_effect = RuntimeError('Simulated agent failure')

            # Call the tool function and expect ModelRetry
            with pytest.raises(ModelRetry, match='failed unexpectedly with error'):
                await tool.function(mock_ctx, 'HelperAgent', 'test input')


class TestMemoryToolSuccessfulAdd:
    """Tests for memory tool successful operations."""

    @pytest.mark.asyncio
    async def test_memory_tool_adds_info_successfully(self, model: TestModel) -> None:
        """Test that adding valid info to memory succeeds."""
        from pydantic_ai.models.test import TestModel as TM
        from pydantic_collab._types import AgentMemory

        test_model = TM(custom_output_text='Result')
        agent = Agent(test_model, name='MemSuccessAgent')
        agent_mem = AgentMemory(name='TestMemory')

        collab = StarCollab(agents=[(agent, 'A')])
        context_list: list[str] = []
        tool = collab._get_add_to_memory_tool(agent_mem, context_list)

        from pydantic_ai import RunContext
        from unittest.mock import MagicMock

        mock_ctx = MagicMock(spec=RunContext)
        result = await tool.function(mock_ctx, 'Valid info')  # Valid info

        # Should return None on success
        assert result is None
        # Should have added to context
        assert 'Valid info' in context_list


class TestMemoryToolValidation:
    """Tests for memory tool validation."""

    @pytest.mark.asyncio
    async def test_memory_tool_empty_info_returns_error(self, model: TestModel) -> None:
        """Test that adding empty info to memory returns error message."""
        from pydantic_ai.models.test import TestModel as TM
        from pydantic_collab._types import AgentMemory

        test_model = TM(custom_output_text='Result')
        agent = Agent(test_model, name='MemAgent')
        agent_mem = AgentMemory(name='TestMemory')

        collab = StarCollab(agents=[(agent, 'A')])
        context_list: list[str] = []
        tool = collab._get_add_to_memory_tool(agent_mem, context_list)

        # Call the tool function directly
        from pydantic_ai import RunContext
        from unittest.mock import MagicMock

        mock_ctx = MagicMock(spec=RunContext)
        result = await tool.function(mock_ctx, '')  # Empty info

        assert result == "Info can't be empty."

    @pytest.mark.asyncio
    async def test_memory_tool_too_long_info_returns_error(self, model: TestModel) -> None:
        """Test that adding too long info to memory returns error message."""
        from pydantic_ai.models.test import TestModel as TM
        from pydantic_collab._types import AgentMemory, MAXIMUM_MEM_LINE_LENGTH

        test_model = TM(custom_output_text='Result')
        agent = Agent(test_model, name='MemLongAgent')
        agent_mem = AgentMemory(name='TestMemory')

        collab = StarCollab(agents=[(agent, 'A')])
        context_list: list[str] = []
        tool = collab._get_add_to_memory_tool(agent_mem, context_list)

        from pydantic_ai import RunContext
        from unittest.mock import MagicMock

        mock_ctx = MagicMock(spec=RunContext)
        long_info = 'x' * (MAXIMUM_MEM_LINE_LENGTH + 1)
        result = await tool.function(mock_ctx, long_info)

        assert f"Info can't be longer than {MAXIMUM_MEM_LINE_LENGTH}" in result
