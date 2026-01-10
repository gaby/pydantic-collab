"""Tests for handoff control and data passing between agents.

These tests focus on realistic scenarios for controlling what data
is passed during handoffs and how context is managed.
"""

import pytest
from pydantic_ai.models.test import TestModel

from pydantic_collab import CollabAgent, PipelineCollab
from pydantic_collab._types import HandOffBase
from tests.test_handoff_tool_control import make_test_agent


@pytest.mark.asyncio
async def test_handoff_data_payload_structure():
    """Test that handoff data payload is correctly structured."""
    model1 = TestModel(
        call_tools=[],
        custom_output_args=HandOffBase(
            next_agent='Analyzer',
            reasoning='Data ready for analysis',
            query='Analyze the processed results',
        ),
    )
    model2 = TestModel(call_tools=[], custom_output_text='Analysis complete')

    processor = make_test_agent('Processor', model1)
    analyzer = make_test_agent('Analyzer', model2)

    swarm = PipelineCollab(
        agents=[
            CollabAgent(agent=processor, description='Processes data', agent_handoffs=('Analyzer',)),
            CollabAgent(agent=analyzer, description='Analyzes results'),
        ],
        starting_agent=processor,
        max_handoffs=3,
    )

    result = await swarm.run('Process this data')

    # Verify execution history structure
    assert len(result.execution_history) == 2

    first_step = result.execution_history[0]
    assert first_step['agent'] == 'Processor'
    assert first_step['action'] == 'handoff'
    assert first_step['next_agent'] == 'Analyzer'
    assert first_step['reasoning'] == 'Data ready for analysis'

    second_step = result.execution_history[1]
    assert second_step['agent'] == 'Analyzer'
    assert second_step['action'] == 'final'
    assert second_step['next_agent'] is None


@pytest.mark.asyncio
async def test_handoff_query_transformation():
    """Test that query can be transformed during handoff."""
    model1 = TestModel(
        call_tools=[],
        custom_output_args=HandOffBase(
            next_agent='Formatter',
            reasoning='Data processed',
            query='Format this output: {"data": "processed"}',  # Transformed query
        ),
    )
    model2 = TestModel(call_tools=[], custom_output_text='Formatted successfully')

    processor = make_test_agent('Processor', model1)
    formatter = make_test_agent('Formatter', model2)

    swarm = PipelineCollab(
        agents=[
            CollabAgent(agent=processor, description='Processes data', agent_handoffs=('Formatter',)),
            CollabAgent(agent=formatter, description='Formats output'),
        ],
        starting_agent=processor,
        max_handoffs=3,
    )

    result = await swarm.run('Original query here')

    # Check that the query was transformed in the handoff
    first_step = result.execution_history[0]
    assert 'Format this output' in first_step['output']

    second_step = result.execution_history[1]
    assert 'Format this output' in second_step['input']


@pytest.mark.asyncio
async def test_handoff_reasoning_preserved():
    """Test that reasoning is preserved in execution history."""
    reasoning_text = 'Complex analysis required, escalating to specialist'

    model1 = TestModel(
        call_tools=[],
        custom_output_args=HandOffBase(
            next_agent='Specialist',
            reasoning=reasoning_text,
            query='Perform deep analysis',
        ),
    )
    model2 = TestModel(call_tools=[], custom_output_text='Deep analysis complete')

    generalist = make_test_agent('Generalist', model1)
    specialist = make_test_agent('Specialist', model2)

    swarm = PipelineCollab(
        agents=[
            CollabAgent(agent=generalist, description='General processor', agent_handoffs=('Specialist',)),
            CollabAgent(agent=specialist, description='Specialist analyzer'),
        ],
        starting_agent=generalist,
        max_handoffs=3,
    )

    result = await swarm.run('Analyze this complex data')

    # Verify reasoning is captured
    first_step = result.execution_history[0]
    assert first_step['reasoning'] == reasoning_text


@pytest.mark.asyncio
async def test_multi_step_data_flow():
    """Test data flowing through multiple agents with transformations."""
    # Simulate a data pipeline: Collector → Processor → Analyzer
    model1 = TestModel(
        call_tools=[],
        custom_output_args=HandOffBase(
            next_agent='Processor',
            reasoning='Data collected',
            query='Process collected data: [1, 2, 3, 4, 5]',
        ),
    )
    model2 = TestModel(
        call_tools=[],
        custom_output_args=HandOffBase(
            next_agent='Analyzer',
            reasoning='Data processed',
            query='Analyze results: sum=15, count=5',
        ),
    )
    model3 = TestModel(call_tools=[], custom_output_text='Analysis: Average is 3.0')

    collector = make_test_agent('Collector', model1)
    processor = make_test_agent('Processor', model2)
    analyzer = make_test_agent('Analyzer', model3)

    swarm = PipelineCollab(
        agents=[
            CollabAgent(agent=collector, description='Collects data', agent_handoffs=('Processor',)),
            CollabAgent(agent=processor, description='Processes data', agent_handoffs=('Analyzer',)),
            CollabAgent(agent=analyzer, description='Analyzes results'),
        ],
        starting_agent=collector,
        max_handoffs=5,
    )

    result = await swarm.run('Collect and analyze data')

    # Verify the data transformation chain
    assert len(result.execution_history) == 3
    assert result.execution_path == ['Collector', 'Processor', 'Analyzer']

    # Check data flows correctly
    step1 = result.execution_history[0]
    assert '[1, 2, 3, 4, 5]' in step1['output']

    step2 = result.execution_history[1]
    assert 'sum=15' in step2['output']

    step3 = result.execution_history[2]
    assert 'Average' in step3['output']


@pytest.mark.asyncio
async def test_conditional_handoff_verification():
    """Test that handoff target choice is verified against allowed targets."""
    # This test verifies the system checks handoff targets
    model1 = TestModel(
        call_tools=[],
        custom_output_args=HandOffBase(
            next_agent='Target2',  # Will hand off to second agent
            reasoning='Routing to second target',
            query='Handle this',
        ),
    )
    model2 = TestModel(custom_output_text='Handled')

    agent1 = make_test_agent('Agent1', model1)
    agent2 = make_test_agent('Target2', model2)

    # In PipelineCollab, agents are chained in order
    swarm = PipelineCollab(
        agents=[
            CollabAgent(agent=agent1, description='First', agent_handoffs=('Target2',)),
            CollabAgent(agent=agent2, description='Second'),
        ],
        starting_agent=agent1,
        max_handoffs=3,
    )

    result = await swarm.run('Test routing')

    # Verify handoff worked
    assert len(result.execution_path) == 2
    assert result.execution_path == ['Agent1', 'Target2']
    assert result.final_agent == 'Target2'


@pytest.mark.asyncio
async def test_handoff_with_empty_reasoning():
    """Test handoff works even with empty reasoning."""
    model1 = TestModel(
        call_tools=[],
        custom_output_args=HandOffBase(
            next_agent='Next',
            reasoning='',  # Empty reasoning
            query='Continue processing',
        ),
    )
    model2 = TestModel(call_tools=[], custom_output_text='Done')

    first = make_test_agent('First', model1)
    second = make_test_agent('Next', model2)

    swarm = PipelineCollab(
        agents=[
            CollabAgent(agent=first, description='First agent', agent_handoffs=('Next',)),
            CollabAgent(agent=second, description='Next agent'),
        ],
        starting_agent=first,
        max_handoffs=3,
    )

    result = await swarm.run('Start')

    # Should still work with empty reasoning
    assert len(result.execution_path) == 2
    first_step = result.execution_history[0]
    assert first_step['reasoning'] == ''


@pytest.mark.asyncio
async def test_handoff_preserves_execution_path():
    """Test that execution path is correctly tracked through handoffs."""
    model1 = TestModel(call_tools=[], custom_output_args=HandOffBase(next_agent='B', reasoning='', query='to B'))
    model2 = TestModel(call_tools=[], custom_output_args=HandOffBase(next_agent='C', reasoning='', query='to C'))
    model3 = TestModel(call_tools=[], custom_output_args=HandOffBase(next_agent='D', reasoning='', query='to D'))
    model4 = TestModel(custom_output_text='final')

    a = make_test_agent('A', model1)
    b = make_test_agent('B', model2)
    c = make_test_agent('C', model3)
    d = make_test_agent('D', model4)

    swarm = PipelineCollab(
        agents=[
            CollabAgent(agent=a, description='A', agent_handoffs=('B',)),
            CollabAgent(agent=b, description='B', agent_handoffs=('C',)),
            CollabAgent(agent=c, description='C', agent_handoffs=('D',)),
            CollabAgent(agent=d, description='D'),
        ],
        starting_agent=a,
        max_handoffs=5,
    )

    result = await swarm.run('test')

    # Verify complete path
    assert result.execution_path == ['A', 'B', 'C', 'D']
    assert len(result.execution_history) == 4
    assert result.final_agent == 'D'
    assert result.iterations == 4  # 4 agent runs (A, B, C, D)


@pytest.mark.asyncio
async def test_handoff_usage_tracking():
    """Test that usage stats are accumulated across handoffs."""
    model1 = TestModel(
        call_tools=[], custom_output_args=HandOffBase(next_agent='Second', reasoning='', query='continue')
    )
    model2 = TestModel(custom_output_text='done')

    first = make_test_agent('First', model1)
    second = make_test_agent('Second', model2)

    swarm = PipelineCollab(
        agents=[
            CollabAgent(agent=first, description='First', agent_handoffs=('Second',)),
            CollabAgent(agent=second, description='Second'),
        ],
        starting_agent=first,
        max_handoffs=3,
    )

    result = await swarm.run('test')

    # Check usage is tracked
    assert result.usage is not None
    assert result.usage.requests >= 2  # At least 2 agent calls


@pytest.mark.asyncio
async def test_handoff_data_isolation():
    """Test that each agent receives only intended data."""
    model1 = TestModel(
        call_tools=[],
        custom_output_args=HandOffBase(
            next_agent='Agent2',
            reasoning='private reasoning',
            query='only this query should pass',
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

    result = await swarm.run('original input')

    # Agent2's input should be the handoff query, not original
    step2 = result.execution_history[1]
    assert 'only this query should pass' in step2['input']


@pytest.mark.asyncio
async def test_long_handoff_chain_data_integrity():
    """Test data integrity through a long chain of handoffs."""
    # Create a 5-agent chain
    models = []
    for i in range(4):
        models.append(
            TestModel(
                call_tools=[],
                custom_output_args=HandOffBase(
                    next_agent=f'Agent{i + 2}',
                    reasoning=f'Step {i + 1} complete',
                    query=f'data_v{i + 1}',
                ),
            )
        )
    models.append(TestModel(custom_output_text='final_result'))

    agents_list = []
    for i in range(5):
        agent = make_test_agent(f'Agent{i + 1}', models[i])
        handoffs = (f'Agent{i + 2}',) if i < 4 else ()
        agents_list.append(CollabAgent(agent=agent, description=f'Agent {i + 1}', agent_handoffs=handoffs))

    swarm = PipelineCollab(
        agents=agents_list,
        starting_agent=agents_list[0].agent,
        max_handoffs=10,
    )

    result = await swarm.run('start')

    # Verify all agents executed in order
    assert len(result.execution_path) == 5
    assert result.execution_path == ['Agent1', 'Agent2', 'Agent3', 'Agent4', 'Agent5']

    # Verify data passed through correctly
    for i in range(4):
        step = result.execution_history[i]
        assert step['reasoning'] == f'Step {i + 1} complete'
        assert f'data_v{i + 1}' in step['output']
