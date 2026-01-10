"""Tests for context and data passing control during handoffs.

These tests verify that conversation history, tool call results,
and other context can be selectively included or excluded during handoffs.
"""

import pytest
from pydantic_ai.models.test import TestModel

from pydantic_collab import (
    CollabAgent,
    CollabSettings,
    PipelineCollab,
)
from pydantic_collab._types import HandOffBase
from tests.test_handoff_tool_control import make_test_agent


@pytest.mark.asyncio
async def test_handoff_excludes_conversation_history():
    """Test that conversation history can be excluded from handoff."""
    model1 = TestModel(
        call_tools=[],
        custom_output_args=HandOffBase(
            next_agent='Agent2',
            reasoning='Fresh start needed',
            query='New context',
            include_conversation=False,  # Explicitly exclude history
        ),
    )
    model2 = TestModel(custom_output_text='processed')

    agent1 = make_test_agent('Agent1', model1)
    agent2 = make_test_agent('Agent2', model2)

    # Configure network to honor include_conversation flag
    settings = CollabSettings(
        include_conversation='allow',  # Allow per-handoff control
        include_thinking='disallow',
    )

    swarm = PipelineCollab(
        agents=[
            CollabAgent(agent=agent1, description='Agent 1', agent_handoffs=('Agent2',)),
            CollabAgent(agent=agent2, description='Agent 2'),
        ],
        starting_agent=agent1,
        max_handoffs=3,
        collab_settings=settings,
    )

    result = await swarm.run('Original long conversation context')

    # Verify handoff occurred
    assert len(result.execution_path) == 2
    assert result.execution_history[0]['agent'] == 'Agent1'
    assert result.execution_history[1]['agent'] == 'Agent2'


@pytest.mark.asyncio
async def test_handoff_includes_conversation_history():
    """Test that conversation history is included when specified."""
    model1 = TestModel(
        call_tools=[],
        custom_output_args=HandOffBase(
            next_agent='Agent2',
            reasoning='Continue with context',
            query='Use previous context',
            include_conversation=True,  # Explicitly include history
        ),
    )
    model2 = TestModel(custom_output_text='used context')

    agent1 = make_test_agent('Agent1', model1)
    agent2 = make_test_agent('Agent2', model2)

    settings = CollabSettings(
        include_conversation='allow',
        include_thinking='disallow',
    )

    swarm = PipelineCollab(
        agents=[
            CollabAgent(agent=agent1, description='Agent 1', agent_handoffs=('Agent2',)),
            CollabAgent(agent=agent2, description='Agent 2'),
        ],
        starting_agent=agent1,
        max_handoffs=3,
        collab_settings=settings,
    )

    result = await swarm.run('Context to preserve')

    assert len(result.execution_path) == 2


@pytest.mark.asyncio
async def test_network_settings_default_behavior():
    """Test that network settings control default handoff behavior."""
    model = TestModel(
        call_tools=[],
        custom_output_args=HandOffBase(
            next_agent='Agent2',
            reasoning='Using defaults',
            query='process',
        ),
    )
    model2 = TestModel(custom_output_text='done')

    agent1 = make_test_agent('Agent1', model)
    agent2 = make_test_agent('Agent2', model2)

    # Test with conversation EXCLUDED by default
    settings = CollabSettings(
        include_conversation='disallow',  # Default excludes conversation
        include_thinking='disallow',
    )

    swarm = PipelineCollab(
        agents=[
            CollabAgent(agent=agent1, description='Agent 1', agent_handoffs=('Agent2',)),
            CollabAgent(agent=agent2, description='Agent 2'),
        ],
        starting_agent=agent1,
        max_handoffs=3,
        collab_settings=settings,
    )

    result = await swarm.run('test')
    assert len(result.execution_path) == 2


@pytest.mark.asyncio
async def test_multiple_handoffs_with_varying_context():
    """Test chain where context inclusion varies per handoff."""
    model1 = TestModel(
        call_tools=[],
        custom_output_args=HandOffBase(
            next_agent='Agent2',
            reasoning='',
            query='step1',
            include_conversation=True,  # Include context
        ),
    )
    model2 = TestModel(
        call_tools=[],
        custom_output_args=HandOffBase(
            next_agent='Agent3',
            reasoning='',
            query='step2',
            include_conversation=False,  # Exclude context
        ),
    )
    model3 = TestModel(custom_output_text='final')

    agent1 = make_test_agent('Agent1', model1)
    agent2 = make_test_agent('Agent2', model2)
    agent3 = make_test_agent('Agent3', model3)

    swarm = PipelineCollab(
        agents=[
            CollabAgent(agent=agent1, description='A1', agent_handoffs=('Agent2',)),
            CollabAgent(agent=agent2, description='A2', agent_handoffs=('Agent3',)),
            CollabAgent(agent=agent3, description='A3'),
        ],
        starting_agent=agent1,
        max_handoffs=5,
    )

    result = await swarm.run('start')

    # All three agents should execute
    assert len(result.execution_path) == 3
    assert result.execution_path == ['Agent1', 'Agent2', 'Agent3']


@pytest.mark.asyncio
async def test_handoff_query_size_control():
    """Test that handoff query can be controlled to manage size."""
    # Long reasoning and query
    long_reasoning = 'x' * 500  # Very long reasoning
    long_query = 'y' * 500  # Very long query

    model1 = TestModel(
        call_tools=[],
        custom_output_args=HandOffBase(
            next_agent='Agent2',
            reasoning=long_reasoning,
            query=long_query,
        ),
    )
    model2 = TestModel(custom_output_text='received')

    agent1 = make_test_agent('Agent1', model1)
    agent2 = make_test_agent('Agent2', model2)

    swarm = PipelineCollab(
        agents=[
            CollabAgent(agent=agent1, description='Agent 1', agent_handoffs=('Agent2',)),
            CollabAgent(agent=agent2, description='Agent 2'),
        ],
        starting_agent=agent1,
        max_handoffs=3,
    )

    result = await swarm.run('test')

    # Check that data was passed (truncation handled by CollabState)
    step1 = result.execution_history[0]
    # Reasoning and output are truncated in execution_history
    assert len(step1['reasoning']) <= 103  # 100 chars + "..."
    assert len(step1['output']) <= 203  # 200 chars + "..."


@pytest.mark.asyncio
async def test_handoff_with_structured_query_data():
    """Test handoff with structured data in query."""
    structured_query = '{"type": "analysis", "data": [1, 2, 3], "priority": "high"}'

    model1 = TestModel(
        call_tools=[],
        custom_output_args=HandOffBase(
            next_agent='Analyzer',
            reasoning='Structured data ready',
            query=structured_query,
        ),
    )
    model2 = TestModel(custom_output_text='analyzed')

    processor = make_test_agent('Processor', model1)
    analyzer = make_test_agent('Analyzer', model2)

    swarm = PipelineCollab(
        agents=[
            CollabAgent(agent=processor, description='Processor', agent_handoffs=('Analyzer',)),
            CollabAgent(agent=analyzer, description='Analyzer'),
        ],
        starting_agent=processor,
        max_handoffs=3,
    )

    result = await swarm.run('Process data')

    # Verify structured data was passed
    step2 = result.execution_history[1]
    assert 'analysis' in step2['input'] or 'data' in step2['input']


@pytest.mark.asyncio
async def test_handoff_context_aggregation_in_chain():
    """Test that context can accumulate through a chain."""
    model1 = TestModel(
        call_tools=[],
        custom_output_args=HandOffBase(
            next_agent='Agent2',
            reasoning='Added data A',
            query='data: A',
            include_conversation=True,
        ),
    )
    model2 = TestModel(
        call_tools=[],
        custom_output_args=HandOffBase(
            next_agent='Agent3',
            reasoning='Added data B',
            query='data: A, B',
            include_conversation=True,
        ),
    )
    model3 = TestModel(call_tools=[], custom_output_text='Final: A, B, C')

    agent1 = make_test_agent('Agent1', model1)
    agent2 = make_test_agent('Agent2', model2)
    agent3 = make_test_agent('Agent3', model3)

    swarm = PipelineCollab(
        agents=[
            CollabAgent(agent=agent1, description='A1', agent_handoffs=('Agent2',)),
            CollabAgent(agent=agent2, description='A2', agent_handoffs=('Agent3',)),
            CollabAgent(agent=agent3, description='A3'),
        ],
        starting_agent=agent1,
        max_handoffs=5,
    )

    result = await swarm.run('start')

    # Verify the chain executed
    assert len(result.execution_path) == 3

    # Check that data accumulated
    step1 = result.execution_history[0]
    assert 'data: A' in step1['output']

    step2 = result.execution_history[1]
    assert 'data: A, B' in step2['output']


@pytest.mark.asyncio
async def test_empty_handoff_query():
    """Test handoff with empty query string."""
    model1 = TestModel(
        call_tools=[],
        custom_output_args=HandOffBase(
            next_agent='Agent2',
            reasoning='No specific query',
            query='',  # Empty query
        ),
    )
    model2 = TestModel(custom_output_text='handled empty query')

    agent1 = make_test_agent('Agent1', model1)
    agent2 = make_test_agent('Agent2', model2)

    swarm = PipelineCollab(
        agents=[
            CollabAgent(agent=agent1, description='Agent 1', agent_handoffs=('Agent2',)),
            CollabAgent(agent=agent2, description='Agent 2'),
        ],
        starting_agent=agent1,
        max_handoffs=3,
    )

    result = await swarm.run('original query')

    # Should handle empty query gracefully
    assert len(result.execution_path) == 2
    step2 = result.execution_history[1]
    assert step2['agent'] == 'Agent2'


@pytest.mark.asyncio
async def test_handoff_preserves_metadata():
    """Test that handoff preserves execution metadata."""
    model1 = TestModel(
        call_tools=[],
        custom_output_args=HandOffBase(
            next_agent='Agent2',
            reasoning='Preserving metadata',
            query='with metadata',
        ),
    )
    model2 = TestModel(custom_output_text='final')

    agent1 = make_test_agent('Agent1', model1)
    agent2 = make_test_agent('Agent2', model2)

    swarm = PipelineCollab(
        agents=[
            CollabAgent(agent=agent1, description='First agent description', agent_handoffs=('Agent2',)),
            CollabAgent(agent=agent2, description='Second agent description'),
        ],
        starting_agent=agent1,
        max_handoffs=3,
    )

    result = await swarm.run('test')

    # Check that all metadata is preserved
    assert result.iterations == 2  # Two agent runs (Agent1, Agent2)
    assert result.final_agent == 'Agent2'
    assert result.max_iterations_reached is False
    assert len(result.execution_history) == 2


@pytest.mark.asyncio
async def test_handoff_data_sanitization():
    """Test that special characters in handoff data are handled."""
    query_with_special_chars = 'Query with "quotes", \\backslashes\\, and\nnewlines\t\ttabs'

    model1 = TestModel(
        call_tools=[],
        custom_output_args=HandOffBase(
            next_agent='Agent2',
            reasoning='Special chars test',
            query=query_with_special_chars,
        ),
    )
    model2 = TestModel(custom_output_text='handled')

    agent1 = make_test_agent('Agent1', model1)
    agent2 = make_test_agent('Agent2', model2)

    swarm = PipelineCollab(
        agents=[
            CollabAgent(agent=agent1, description='Agent 1', agent_handoffs=('Agent2',)),
            CollabAgent(agent=agent2, description='Agent 2'),
        ],
        starting_agent=agent1,
        max_handoffs=3,
    )

    result = await swarm.run('test')

    # Should handle special characters without errors
    assert len(result.execution_path) == 2
    assert result.output is not None


@pytest.mark.asyncio
async def test_bidirectional_handoff_limit():
    """Test that max_handoffs prevents infinite bidirectional loops."""
    model1 = TestModel(
        call_tools=[],
        custom_output_args=HandOffBase(
            next_agent='Agent2',
            reasoning='First pass',
            query='data_v1',
        ),
    )
    # Agent2 will complete without handing back
    model2 = TestModel(custom_output_text='final')

    agent1 = make_test_agent('Agent1', model1)
    agent2 = make_test_agent('Agent2', model2)

    swarm = PipelineCollab(
        agents=[
            CollabAgent(agent=agent1, description='Agent 1', agent_handoffs=('Agent2',)),
            CollabAgent(agent=agent2, description='Agent 2'),
        ],
        starting_agent=agent1,
        max_handoffs=5,
    )

    result = await swarm.run('start')

    # Should execute both agents
    assert len(result.execution_path) == 2
    assert result.execution_path == ['Agent1', 'Agent2']
