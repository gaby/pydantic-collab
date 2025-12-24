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
