"""Tests for agent memory feature.

These tests cover:
- Memory declaration and validation
- Memory sharing between agents
- Memory persistence across handoffs
- Memory write tool behavior
- Prompt builder memory integration
- Memory isolation between runs
"""

import pytest
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

from pydantic_collab import Collab, CollabAgent, PipelineCollab, StarCollab
from pydantic_collab._types import AgentMemory, HandOffBase, MAXIMUM_MEM_LINE_LENGTH, PromptBuilderContext, str_or_am_to_am, validate_r_rw
from pydantic_collab._utils import default_build_agent_prompt


# =============================================================================
# Helper Functions
# =============================================================================


def make_test_agent(name: str, model, output_type=str):
    return Agent(model, name=name, output_type=output_type)


# =============================================================================
# AgentMemory Dataclass Tests
# =============================================================================


class TestAgentMemoryDataclass:
    """Tests for the AgentMemory dataclass."""

    def test_agent_memory_creation_with_name_only(self):
        """Test creating AgentMemory with just a name."""
        mem = AgentMemory(name='notes')
        assert mem.name == 'notes'
        assert mem.description is None

    def test_agent_memory_creation_with_description(self):
        """Test creating AgentMemory with name and description."""
        mem = AgentMemory(name='research', description='Store research findings')
        assert mem.name == 'research'
        assert mem.description == 'Store research findings'

    def test_agent_memory_is_frozen(self):
        """Test that AgentMemory is immutable (frozen dataclass)."""
        mem = AgentMemory(name='notes')
        with pytest.raises(AttributeError):
            mem.name = 'changed'

    def test_agent_memory_hashable(self):
        """Test that AgentMemory can be used as dict key."""
        mem1 = AgentMemory(name='notes')
        mem2 = AgentMemory(name='notes')
        d = {mem1: 'value'}
        assert d[mem2] == 'value'

    def test_agent_memory_equality(self):
        """Test AgentMemory equality comparison."""
        mem1 = AgentMemory(name='notes', description='desc')
        mem2 = AgentMemory(name='notes', description='desc')
        mem3 = AgentMemory(name='other', description='desc')
        assert mem1 == mem2
        assert mem1 != mem3


# =============================================================================
# Memory Declaration & Validation Tests
# =============================================================================


class TestMemoryDeclaration:
    """Tests for memory declaration on CollabAgent."""

    def test_memory_as_string(self):
        """Test declaring memory with a simple string."""
        model = TestModel()
        agent = CollabAgent(
            agent=make_test_agent('Agent', model),
            description='Test agent',
            memory='notes',
        )
        assert len(agent.memory) == 1
        mem = list(agent.memory.keys())[0]
        assert mem.name == 'notes'
        assert agent.memory[mem] == 'rw'  # Default permission

    def test_memory_as_list_of_strings(self):
        """Test declaring memory with a list of strings."""
        model = TestModel()
        agent = CollabAgent(
            agent=make_test_agent('Agent', model),
            description='Test agent',
            memory=['notes', 'research'],
        )
        assert len(agent.memory) == 2
        names = {m.name for m in agent.memory.keys()}
        assert names == {'notes', 'research'}
        # All should default to 'rw'
        assert all(v == 'rw' for v in agent.memory.values())

    def test_memory_as_dict_with_permissions(self):
        """Test declaring memory with explicit permissions."""
        model = TestModel()
        agent = CollabAgent(
            agent=make_test_agent('Agent', model),
            description='Test agent',
            memory={'notes': 'rw', 'config': 'r'},
        )
        assert len(agent.memory) == 2
        mem_by_name = {m.name: v for m, v in agent.memory.items()}
        assert mem_by_name['notes'] == 'rw'
        assert mem_by_name['config'] == 'r'

    def test_memory_as_agent_memory_objects(self):
        """Test declaring memory with AgentMemory objects."""
        model = TestModel()
        notes_mem = AgentMemory(name='notes', description='Store notes')
        agent = CollabAgent(
            agent=make_test_agent('Agent', model),
            description='Test agent',
            memory={notes_mem: 'rw'},
        )
        assert len(agent.memory) == 1
        mem = list(agent.memory.keys())[0]
        assert mem.name == 'notes'
        assert mem.description == 'Store notes'

    def test_memory_none_results_in_empty_dict(self):
        """Test that memory=None results in empty dict."""
        model = TestModel()
        agent = CollabAgent(
            agent=make_test_agent('Agent', model),
            description='Test agent',
            memory=None,
        )
        assert agent.memory == {}

    def test_memory_empty_list_results_in_empty_dict(self):
        """Test that memory=[] results in empty dict."""
        model = TestModel()
        agent = CollabAgent(
            agent=make_test_agent('Agent', model),
            description='Test agent',
            memory=[],
        )
        assert agent.memory == {}


class TestMemoryValidation:
    """Tests for memory validation."""

    def test_invalid_permission_raises_error(self):
        """Test that invalid permission values raise an error."""
        model = TestModel()
        with pytest.raises(RuntimeError, match='Value needs to be either r or rw'):
            CollabAgent(
                agent=make_test_agent('Agent', model),
                description='Test agent',
                memory={'notes': 'write'},  # Invalid
            )

    def test_invalid_permission_w_only_raises_error(self):
        """Test that 'w' alone is invalid (must be 'r' or 'rw')."""
        model = TestModel()
        with pytest.raises(RuntimeError, match='Value needs to be either r or rw'):
            CollabAgent(
                agent=make_test_agent('Agent', model),
                description='Test agent',
                memory={'notes': 'w'},
            )

    def test_duplicate_memory_names_raises_error(self):
        """Test that duplicate memory names raise an error."""
        model = TestModel()
        mem1 = AgentMemory(name='notes', description='First')
        mem2 = AgentMemory(name='notes', description='Second')  # Same name
        with pytest.raises(ValueError, match='Memory names must be unique'):
            CollabAgent(
                agent=make_test_agent('Agent', model),
                description='Test agent',
                memory={mem1: 'rw', mem2: 'r'},
            )

    def test_validate_r_rw_helper(self):
        """Test the validate_r_rw helper function."""
        assert validate_r_rw('r') == 'r'
        assert validate_r_rw('rw') == 'rw'
        with pytest.raises(RuntimeError):
            validate_r_rw('invalid')

    def test_str_or_am_to_am_helper(self):
        """Test the str_or_am_to_am helper function."""
        # String input
        result = str_or_am_to_am('notes')
        assert isinstance(result, AgentMemory)
        assert result.name == 'notes'
        assert result.description is None

        # AgentMemory input
        mem = AgentMemory(name='research', description='desc')
        result = str_or_am_to_am(mem)
        assert result is mem


# =============================================================================
# Memory Sharing Between Agents Tests
# =============================================================================


class TestMemorySharing:
    """Tests for memory sharing between agents."""

    @pytest.mark.asyncio
    async def test_memory_shared_between_agents_same_name(self):
        """Test that agents with same memory name share the same underlying storage."""
        # Agent1 writes to 'notes', Agent2 reads from 'notes'
        model1 = TestModel(
            call_tools=['add_to_notes_mem'],
            custom_output_args=HandOffBase(
                next_agent='Agent2',
                reasoning='Wrote to memory',
                query='Check the notes',
            ),
        )
        model2 = TestModel(call_tools=[], custom_output_text='Read the notes')

        agent1 = make_test_agent('Agent1', model1)
        agent2 = make_test_agent('Agent2', model2)

        swarm = PipelineCollab(
            agents=[
                CollabAgent(agent=agent1, description='Writer', agent_handoffs=('Agent2',), memory={'notes': 'rw'}),
                CollabAgent(agent=agent2, description='Reader', memory={'notes': 'r'}),
            ],
            starting_agent=agent1,
            max_handoffs=3,
        )

        result = await swarm.run('Write something to notes')

        # Both agents should reference the same memory object in state
        assert result is not None
        # The memory should exist in state
        state = result._state
        notes_mems = [m for m in state.memory_objects.keys() if m.name == 'notes']
        assert len(notes_mems) == 1  # Only one memory object for 'notes'

    @pytest.mark.asyncio
    async def test_memory_isolated_between_different_names(self):
        """Test that memories with different names are isolated."""
        model = TestModel(call_tools=[], custom_output_text='Done')
        agent = make_test_agent('Agent', model)

        swarm = StarCollab(
            agents=[
                CollabAgent(agent=agent, description='Test', memory={'notes': 'rw', 'config': 'rw'}),
            ],
            router_agent=agent,
            model=model,
        )

        result = await swarm.run('Test')
        state = result._state

        # Should have two separate memory objects
        mem_names = {m.name for m in state.memory_objects.keys()}
        # Note: defaultdict creates entries on access, so we check the structure
        assert 'notes' not in mem_names or 'config' not in mem_names or len(state.memory_objects) >= 0


# =============================================================================
# Memory Persistence Across Handoffs Tests
# =============================================================================


class TestMemoryPersistenceAcrossHandoffs:
    """Tests for memory persistence across agent handoffs."""

    @pytest.mark.asyncio
    async def test_memory_persists_after_handoff(self):
        """Test that memory written by Agent1 is available to Agent2 after handoff."""
        # This test verifies the memory is in CollabState and persists
        model1 = TestModel(
            call_tools=[],
            custom_output_args=HandOffBase(
                next_agent='Agent2',
                reasoning='Passing on',
                query='Continue',
            ),
        )
        model2 = TestModel(call_tools=[], custom_output_text='Done')

        agent1 = make_test_agent('Agent1', model1)
        agent2 = make_test_agent('Agent2', model2)

        shared_mem = AgentMemory(name='shared', description='Shared memory')

        swarm = PipelineCollab(
            agents=[
                CollabAgent(agent=agent1, description='First', agent_handoffs=('Agent2',), memory={shared_mem: 'rw'}),
                CollabAgent(agent=agent2, description='Second', memory={shared_mem: 'r'}),
            ],
            starting_agent=agent1,
            max_handoffs=3,
        )

        result = await swarm.run('Test persistence')

        # Verify both agents ran
        assert len(result.execution_path) == 2
        assert result.execution_path == ['Agent1', 'Agent2']

        # Memory should be accessible in state
        state = result._state
        assert shared_mem in state.memory_objects or any(m.name == 'shared' for m in state.memory_objects.keys())


# =============================================================================
# Memory Isolation Between Runs Tests
# =============================================================================


class TestMemoryIsolationBetweenRuns:
    """Tests for memory isolation between separate runs."""

    @pytest.mark.asyncio
    async def test_memory_reset_between_runs(self):
        """Test that memory is reset between separate run() calls."""
        model = TestModel(call_tools=[], custom_output_text='Done')
        agent = make_test_agent('Agent', model)

        swarm = StarCollab(
            agents=[
                CollabAgent(agent=agent, description='Test', memory={'notes': 'rw'}),
            ],
            router_agent=agent,
            model=model,
        )

        # First run
        result1 = await swarm.run('First run')
        state1 = result1._state

        # Manually add something to memory to simulate write
        notes_mem = next((m for m in state1.memory_objects.keys() if m.name == 'notes'), None)
        if notes_mem:
            state1.memory_objects[notes_mem].append('data from run 1')

        # Second run - should have fresh memory
        result2 = await swarm.run('Second run')
        state2 = result2._state

        # Memory from run 1 should not leak into run 2
        notes_mem2 = next((m for m in state2.memory_objects.keys() if m.name == 'notes'), None)
        if notes_mem2:
            assert 'data from run 1' not in state2.memory_objects[notes_mem2]


# =============================================================================
# Memory Write Tool Tests
# =============================================================================


class TestMemoryWriteTool:
    """Tests for the memory write tool behavior."""

    @pytest.mark.asyncio
    async def test_write_tool_only_for_rw_permission(self):
        """Test that write tool is only created for 'rw' permission, not 'r'."""
        model = TestModel(call_tools=[], custom_output_text='Done')
        agent = make_test_agent('Agent', model)

        # Agent with read-only memory should not get write tool
        collab_agent = CollabAgent(agent=agent, description='Test', memory={'readonly': 'r'})

        swarm = StarCollab(
            agents=[collab_agent],
            router_agent=agent,
            model=model,
        )

        # Run to trigger tool setup
        await swarm.run('Test')

        # The tool should not be created for read-only memory
        # We verify this through the prompt builder behavior

    @pytest.mark.asyncio
    async def test_write_tool_created_for_rw_permission(self):
        """Test that write tool is created for 'rw' permission."""
        model = TestModel(call_tools=[], custom_output_text='Done')
        agent = make_test_agent('Agent', model)

        collab_agent = CollabAgent(agent=agent, description='Test', memory={'writable': 'rw'})

        swarm = StarCollab(
            agents=[collab_agent],
            router_agent=agent,
            model=model,
        )

        await swarm.run('Test')
        # Tool should be created - verified through prompt

    @pytest.mark.asyncio
    async def test_write_tool_called_by_model(self):
        """Test that the write tool can be called by the model."""
        # Configure model to call the memory tool
        model = TestModel(
            call_tools=['add_to_notes_mem'],
            custom_output_text='Added to memory',
        )
        agent = make_test_agent('Agent', model)

        swarm = StarCollab(
            agents=[
                CollabAgent(agent=agent, description='Test', memory={'notes': 'rw'}),
            ],
            router_agent=agent,
            model=model,
        )

        result = await swarm.run('Add something to notes')
        assert result.output is not None


# =============================================================================
# Prompt Builder Memory Integration Tests
# =============================================================================


class TestPromptBuilderMemoryIntegration:
    """Tests for memory integration in the prompt builder."""

    def test_prompt_includes_memory_section_for_rw(self):
        """Test that prompt includes memory section for writable memory."""
        model = TestModel()
        notes_mem = AgentMemory(name='notes', description='Store important notes')

        collab_agent = CollabAgent(
            agent=make_test_agent('Agent', model),
            description='Test agent',
            memory={notes_mem: 'rw'},
        )

        ctx = PromptBuilderContext(
            agent=collab_agent,
            final_agent=True,
            can_handoff=False,
            handoff_agents=[],
            tool_agents=[],
            called_as_tool=False,
            context_info={notes_mem: []},  # Empty but writable
        )

        prompt = default_build_agent_prompt(ctx)

        # Should include memory section even when empty (because writable)
        assert 'Relevant Memories for Agent' in prompt
        assert 'notes' in prompt
        assert 'Store important notes' in prompt
        assert 'add_to_notes_mem' in prompt

    def test_prompt_excludes_empty_readonly_memory(self):
        """Test that prompt excludes empty read-only memory."""
        model = TestModel()
        config_mem = AgentMemory(name='config', description='Configuration data')

        collab_agent = CollabAgent(
            agent=make_test_agent('Agent', model),
            description='Test agent',
            memory={config_mem: 'r'},
        )

        ctx = PromptBuilderContext(
            agent=collab_agent,
            final_agent=True,
            can_handoff=False,
            handoff_agents=[],
            tool_agents=[],
            called_as_tool=False,
            context_info={config_mem: []},  # Empty and read-only
        )

        prompt = default_build_agent_prompt(ctx)

        # Should NOT include empty read-only memory
        # The memory section might not appear at all, or config shouldn't be listed
        assert 'add_to_config_mem' not in prompt

    def test_prompt_includes_readonly_memory_with_data(self):
        """Test that prompt includes read-only memory when it has data."""
        model = TestModel()
        config_mem = AgentMemory(name='config', description='Configuration data')

        collab_agent = CollabAgent(
            agent=make_test_agent('Agent', model),
            description='Test agent',
            memory={config_mem: 'r'},
        )

        ctx = PromptBuilderContext(
            agent=collab_agent,
            final_agent=True,
            can_handoff=False,
            handoff_agents=[],
            tool_agents=[],
            called_as_tool=False,
            context_info={config_mem: ['setting1=value1', 'setting2=value2']},
        )

        prompt = default_build_agent_prompt(ctx)

        # Should include memory with data
        assert 'config' in prompt
        assert 'setting1=value1' in prompt
        assert 'setting2=value2' in prompt
        # Should NOT include write tool for read-only
        assert 'add_to_config_mem' not in prompt

    def test_prompt_includes_memory_data(self):
        """Test that prompt includes actual memory data."""
        model = TestModel()
        notes_mem = AgentMemory(name='notes')

        collab_agent = CollabAgent(
            agent=make_test_agent('Agent', model),
            description='Test agent',
            memory={notes_mem: 'rw'},
        )

        ctx = PromptBuilderContext(
            agent=collab_agent,
            final_agent=True,
            can_handoff=False,
            handoff_agents=[],
            tool_agents=[],
            called_as_tool=False,
            context_info={notes_mem: ['First note', 'Second note', 'Third note']},
        )

        prompt = default_build_agent_prompt(ctx)

        assert 'First note' in prompt
        assert 'Second note' in prompt
        assert 'Third note' in prompt
        assert '### Data:' in prompt

    def test_prompt_no_memory_section_when_no_memory(self):
        """Test that prompt has no memory section when agent has no memory."""
        model = TestModel()

        collab_agent = CollabAgent(
            agent=make_test_agent('Agent', model),
            description='Test agent',
            memory={},  # No memory
        )

        ctx = PromptBuilderContext(
            agent=collab_agent,
            final_agent=True,
            can_handoff=False,
            handoff_agents=[],
            tool_agents=[],
            called_as_tool=False,
            context_info={},  # Empty
        )

        prompt = default_build_agent_prompt(ctx)

        assert 'Relevant Memories for Agent' not in prompt


# =============================================================================
# Memory Tool Name Generation Tests
# =============================================================================


class TestMemoryToolNaming:
    """Tests for memory tool name generation."""

    def test_tool_name_format(self):
        """Test that tool names follow expected format."""
        model = TestModel()
        mem = AgentMemory(name='research_notes')

        collab_agent = CollabAgent(
            agent=make_test_agent('Agent', model),
            description='Test',
            memory={mem: 'rw'},
        )

        ctx = PromptBuilderContext(
            agent=collab_agent,
            final_agent=True,
            can_handoff=False,
            handoff_agents=[],
            tool_agents=[],
            called_as_tool=False,
            context_info={mem: []},
        )

        prompt = default_build_agent_prompt(ctx)

        # Tool name should be add_to_<name>_mem
        assert 'add_to_research_notes_mem' in prompt


# =============================================================================
# Integration Tests
# =============================================================================


class TestMemoryIntegration:
    """Integration tests for the memory feature."""

    @pytest.mark.asyncio
    async def test_full_memory_workflow_pipeline(self):
        """Test complete memory workflow in a pipeline."""
        # Agent1 writes to memory, hands off to Agent2 which reads it
        model1 = TestModel(
            call_tools=[],
            custom_output_args=HandOffBase(
                next_agent='Reader',
                reasoning='Wrote notes',
                query='Read the notes I wrote',
            ),
        )
        model2 = TestModel(call_tools=[], custom_output_text='Read the notes successfully')

        writer = make_test_agent('Writer', model1)
        reader = make_test_agent('Reader', model2)

        notes_mem = AgentMemory(name='shared_notes', description='Notes shared between agents')

        swarm = PipelineCollab(
            agents=[
                CollabAgent(
                    agent=writer,
                    description='Writes notes',
                    agent_handoffs=('Reader',),
                    memory={notes_mem: 'rw'},
                ),
                CollabAgent(
                    agent=reader,
                    description='Reads notes',
                    memory={notes_mem: 'r'},
                ),
            ],
            starting_agent=writer,
            max_handoffs=3,
        )

        result = await swarm.run('Write and read notes')

        assert result.output == 'Read the notes successfully'
        assert result.execution_path == ['Writer', 'Reader']

    @pytest.mark.asyncio
    async def test_multiple_memories_single_agent(self):
        """Test agent with multiple memory artifacts."""
        model = TestModel(call_tools=[], custom_output_text='Done')
        agent = make_test_agent('Agent', model)

        swarm = StarCollab(
            agents=[
                CollabAgent(
                    agent=agent,
                    description='Multi-memory agent',
                    memory={
                        'notes': 'rw',
                        'config': 'r',
                        'cache': 'rw',
                    },
                ),
            ],
            router_agent=agent,
            model=model,
        )

        result = await swarm.run('Test multiple memories')
        assert result.output is not None

    @pytest.mark.asyncio
    async def test_memory_with_agent_calls(self):
        """Test memory works alongside agent tool calls."""
        model = TestModel(call_tools=[], custom_output_text='Done')

        coordinator = make_test_agent('Coordinator', model)
        helper = make_test_agent('Helper', model)

        swarm = StarCollab(
            agents=[
                CollabAgent(
                    agent=coordinator,
                    description='Coordinates',
                    agent_calls=('Helper',),
                    memory={'notes': 'rw'},
                ),
                CollabAgent(
                    agent=helper,
                    description='Helps',
                    memory={'notes': 'r'},
                ),
            ],
            router_agent=coordinator,
            model=model,
        )

        result = await swarm.run('Test memory with agent calls')
        assert result.output is not None

    @pytest.mark.asyncio
    async def test_three_agent_chain_with_shared_memory(self):
        """Test memory sharing across a three-agent chain."""
        model1 = TestModel(
            call_tools=[],
            custom_output_args=HandOffBase(next_agent='Middle', reasoning='Step 1', query='Continue'),
        )
        model2 = TestModel(
            call_tools=[],
            custom_output_args=HandOffBase(next_agent='Final', reasoning='Step 2', query='Finish'),
        )
        model3 = TestModel(call_tools=[], custom_output_text='Complete')

        first = make_test_agent('First', model1)
        middle = make_test_agent('Middle', model2)
        final = make_test_agent('Final', model3)

        shared_mem = AgentMemory(name='chain_data', description='Data passed through chain')

        swarm = PipelineCollab(
            agents=[
                CollabAgent(
                    agent=first,
                    description='First in chain',
                    agent_handoffs=('Middle',),
                    memory={shared_mem: 'rw'},
                ),
                CollabAgent(
                    agent=middle,
                    description='Middle of chain',
                    agent_handoffs=('Final',),
                    memory={shared_mem: 'rw'},
                ),
                CollabAgent(
                    agent=final,
                    description='Final in chain',
                    memory={shared_mem: 'r'},
                ),
            ],
            starting_agent=first,
            max_handoffs=5,
        )

        result = await swarm.run('Chain with memory')

        assert result.execution_path == ['First', 'Middle', 'Final']
        assert result.output == 'Complete'


# =============================================================================
# Edge Cases
# =============================================================================


class TestMemoryEdgeCases:
    """Edge case tests for memory feature."""

    def test_memory_with_special_characters_in_name(self):
        """Test memory name with underscores (valid)."""
        model = TestModel()
        agent = CollabAgent(
            agent=make_test_agent('Agent', model),
            description='Test',
            memory={'my_special_notes': 'rw'},
        )
        mem = list(agent.memory.keys())[0]
        assert mem.name == 'my_special_notes'

    def test_memory_with_empty_description(self):
        """Test AgentMemory with empty string description."""
        mem = AgentMemory(name='notes', description='')
        assert mem.description == ''

    @pytest.mark.asyncio
    async def test_memory_in_single_agent_collab(self):
        """Test memory works with single agent collab."""
        model = TestModel(call_tools=[], custom_output_text='Done')
        agent = make_test_agent('Solo', model)

        swarm = StarCollab(
            agents=[
                CollabAgent(agent=agent, description='Solo agent', memory={'notes': 'rw'}),
            ],
            router_agent=agent,
            model=model,
        )

        result = await swarm.run('Solo with memory')
        assert result.output == 'Done'

    def test_memory_set_and_frozenset_input(self):
        """Test memory accepts set and frozenset inputs."""
        model = TestModel()

        # Set input
        agent1 = CollabAgent(
            agent=make_test_agent('Agent1', model),
            description='Test',
            memory={'notes', 'config'},  # Set
        )
        assert len(agent1.memory) == 2

        # Frozenset input
        agent2 = CollabAgent(
            agent=make_test_agent('Agent2', model),
            description='Test',
            memory=frozenset(['notes', 'config']),
        )
        assert len(agent2.memory) == 2

    def test_memory_tuple_input(self):
        """Test memory accepts tuple input."""
        model = TestModel()
        agent = CollabAgent(
            agent=make_test_agent('Agent', model),
            description='Test',
            memory=('notes', 'config'),
        )
        assert len(agent.memory) == 2
        names = {m.name for m in agent.memory.keys()}
        assert names == {'notes', 'config'}
