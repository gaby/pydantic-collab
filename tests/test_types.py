"""Unit tests for pydantic_collab._types module.

Tests type definitions, settings classes, and data structures.
"""

import pytest
from pydantic_ai import Agent, RunUsage

from pydantic_collab import CollabAgent, CollabError, CollabState
from pydantic_collab._types import (
    AgentContext,
    AgentRunSummary,
    CollabRunResult,
    CollabSettings,
    HandOffBase,
    HandoffData,
    PromptBuilderContext,
    ensure_tuple,
    get_right_handoff_model,
)


class TestCollabSettings:
    """Tests for CollabSettings dataclass."""

    def test_default_settings(self):
        settings = CollabSettings()
        assert settings.include_thinking == 'disallow'
        assert settings.include_conversation == 'allow'
        assert settings.include_handoff == 'allow'
        assert settings.include_tool_calls_with_callee == 'allow'
        assert settings.output_restrictions == 'str_or_original'
        assert settings.include_topology_in_prompt is True
        assert settings.prompt_builder is None
        assert settings.context_builder is None

    def test_custom_settings(self):
        settings = CollabSettings(
            include_thinking='force',
            include_conversation='disallow',
            include_handoff='force',
            output_restrictions='only_str',
        )
        assert settings.include_thinking == 'force'
        assert settings.include_conversation == 'disallow'
        assert settings.include_handoff == 'force'
        assert settings.output_restrictions == 'only_str'

    def test_custom_prompt_builder(self):
        def my_builder(ctx: PromptBuilderContext) -> str:
            return f'Custom prompt for {ctx.agent.name}'

        settings = CollabSettings(prompt_builder=my_builder)
        assert settings.prompt_builder is my_builder

    def test_custom_context_builder(self):
        def my_context(data: HandoffData) -> str:
            return f'Context from {data.caller_agent_name}'

        settings = CollabSettings(context_builder=my_context)
        assert settings.context_builder is my_context


class TestGetRightHandoffModel:
    """Tests for get_right_handoff_model dynamic model generation."""

    def test_default_settings_creates_allow_fields(self):
        settings = CollabSettings()
        Model = get_right_handoff_model(settings)

        # Check model can be instantiated
        instance = Model(next_agent='TestAgent', query='Test query')
        assert instance.next_agent == 'TestAgent'
        assert instance.query == 'Test query'

        # Default values for optional fields
        assert instance.include_conversation is False
        assert instance.include_previous_handoff is False
        assert instance.include_tool_calls_with_callee is False
        assert instance.include_thinking is False

    def test_force_conversation_creates_classvar(self):
        settings = CollabSettings(include_conversation='force')
        Model = get_right_handoff_model(settings)

        instance = Model(next_agent='TestAgent', query='Test')
        # When forced, it's a ClassVar set to True
        assert instance.include_conversation is True

    def test_disallow_conversation_creates_classvar(self):
        settings = CollabSettings(include_conversation='disallow')
        Model = get_right_handoff_model(settings)

        instance = Model(next_agent='TestAgent', query='Test')
        # When disallowed, it's a ClassVar set to False
        assert instance.include_conversation is False

    def test_force_thinking_creates_classvar(self):
        settings = CollabSettings(include_thinking='force')
        Model = get_right_handoff_model(settings)

        instance = Model(next_agent='TestAgent', query='Test')
        assert instance.include_thinking is True

    def test_disallow_thinking_creates_classvar(self):
        settings = CollabSettings(include_thinking='disallow')
        Model = get_right_handoff_model(settings)

        instance = Model(next_agent='TestAgent', query='Test')
        assert instance.include_thinking is False

    def test_force_handoff_creates_classvar(self):
        settings = CollabSettings(include_handoff='force')
        Model = get_right_handoff_model(settings)

        instance = Model(next_agent='TestAgent', query='Test')
        assert instance.include_previous_handoff is True

    def test_force_tool_calls_creates_classvar(self):
        settings = CollabSettings(include_tool_calls_with_callee='force')
        Model = get_right_handoff_model(settings)

        instance = Model(next_agent='TestAgent', query='Test')
        assert instance.include_tool_calls_with_callee is True

    def test_all_force_settings(self):
        settings = CollabSettings(
            include_thinking='force',
            include_conversation='force',
            include_handoff='force',
            include_tool_calls_with_callee='force',
        )
        Model = get_right_handoff_model(settings)

        instance = Model(next_agent='TestAgent', query='Test')
        assert instance.include_thinking is True
        assert instance.include_conversation is True
        assert instance.include_previous_handoff is True
        assert instance.include_tool_calls_with_callee is True

    def test_all_disallow_settings(self):
        settings = CollabSettings(
            include_thinking='disallow',
            include_conversation='disallow',
            include_handoff='disallow',
            include_tool_calls_with_callee='disallow',
        )
        Model = get_right_handoff_model(settings)

        instance = Model(next_agent='TestAgent', query='Test')
        assert instance.include_thinking is False
        assert instance.include_conversation is False
        assert instance.include_previous_handoff is False
        assert instance.include_tool_calls_with_callee is False

    def test_model_is_subclass_of_handoff_base(self):
        settings = CollabSettings()
        Model = get_right_handoff_model(settings)

        assert issubclass(Model, HandOffBase)

    def test_model_with_reasoning(self):
        settings = CollabSettings()
        Model = get_right_handoff_model(settings)

        instance = Model(
            next_agent='TestAgent', query='Test query', reasoning='Because reasons'
        )
        assert instance.reasoning == 'Because reasons'


class TestCollabAgent:
    """Tests for CollabAgent wrapper class."""

    @pytest.fixture
    def base_agent(self):
        return Agent('test', name='TestAgent')

    def test_basic_initialization(self, base_agent):
        collab_agent = CollabAgent(agent=base_agent, description='Test description')
        assert collab_agent.name == 'TestAgent'
        assert collab_agent.description == 'Test description'
        assert collab_agent.agent_calls == ()
        assert collab_agent.agent_handoffs == ()

    def test_single_agent_call_becomes_tuple(self, base_agent):
        other_agent = Agent('test', name='OtherAgent')
        collab_agent = CollabAgent(
            agent=base_agent,
            description='Test',
            agent_calls='OtherAgent',
        )
        assert collab_agent.agent_calls == ('OtherAgent',)

    def test_list_agent_calls_becomes_tuple(self, base_agent):
        collab_agent = CollabAgent(
            agent=base_agent,
            description='Test',
            agent_calls=['Agent1', 'Agent2'],
        )
        assert collab_agent.agent_calls == ('Agent1', 'Agent2')

    def test_single_agent_handoff_becomes_tuple(self, base_agent):
        collab_agent = CollabAgent(
            agent=base_agent,
            description='Test',
            agent_handoffs='NextAgent',
        )
        assert collab_agent.agent_handoffs == ('NextAgent',)

    def test_custom_name_overrides_agent_name(self):
        agent = Agent('test', name='OriginalName')
        collab_agent = CollabAgent(
            agent=agent, description='Test', name='CustomName'
        )
        assert collab_agent.name == 'CustomName'

    def test_agent_without_name_raises_error(self):
        agent = Agent('test')  # No name
        with pytest.raises(ValueError, match='must have a name'):
            CollabAgent(agent=agent, description='Test')

    def test_requires_deps_false_when_no_deps_type(self, base_agent):
        collab_agent = CollabAgent(agent=base_agent, description='Test')
        assert collab_agent.requires_deps is False

    def test_equality_by_name(self):
        agent1 = Agent('test', name='SameName')
        agent2 = Agent('test', name='SameName')

        collab1 = CollabAgent(agent=agent1, description='Desc1')
        collab2 = CollabAgent(agent=agent2, description='Desc2')

        assert collab1 == collab2

    def test_inequality_by_name(self):
        agent1 = Agent('test', name='Name1')
        agent2 = Agent('test', name='Name2')

        collab1 = CollabAgent(agent=agent1, description='Desc')
        collab2 = CollabAgent(agent=agent2, description='Desc')

        assert collab1 != collab2

    def test_hashable(self, base_agent):
        collab_agent = CollabAgent(agent=base_agent, description='Test')
        # Should not raise
        hash(collab_agent)
        # Can be used in sets
        s = {collab_agent}
        assert collab_agent in s

    def test_repr(self, base_agent):
        collab_agent = CollabAgent(agent=base_agent, description='Test description')
        r = repr(collab_agent)
        assert 'CollabAgent' in r
        assert 'TestAgent' in r
        assert 'Test description' in r


class TestCollabState:
    """Tests for CollabState dataclass."""

    def test_initialization(self):
        state = CollabState(query='Test query')
        assert state.query == 'Test query'
        assert state.final_output == ''
        assert state.agent_contexts == {}
        assert state.execution_path == []
        assert state.execution_history == []
        assert state.messages == []

    def test_get_context_creates_new_context(self):
        state = CollabState(query='Test')
        ctx = state.get_context('Agent1')

        assert 'Agent1' in state.agent_contexts
        assert isinstance(ctx, AgentContext)

    def test_get_context_returns_existing(self):
        state = CollabState(query='Test')
        ctx1 = state.get_context('Agent1')
        ctx1.last_output = 'some output'

        ctx2 = state.get_context('Agent1')
        assert ctx2 is ctx1
        assert ctx2.last_output == 'some output'

    def test_record_execution_final_output(self):
        state = CollabState(query='Test')
        state.record_execution('Agent1', 'input query', 'final output')

        assert len(state.execution_history) == 1
        step = state.execution_history[0]
        assert step['agent'] == 'Agent1'
        assert step['action'] == 'final'
        assert step['input'] == 'input query'
        assert step['output'] == 'final output'
        assert step['next_agent'] is None

    def test_record_execution_handoff(self):
        # Create a mock handoff-like object
        class MockHandoff:
            next_agent = 'Agent2'
            query = 'handoff query'
            reasoning = 'because'

        state = CollabState(query='Test')
        state.record_execution('Agent1', 'input', MockHandoff())

        step = state.execution_history[0]
        assert step['action'] == 'handoff'
        assert step['next_agent'] == 'Agent2'
        assert 'because' in step['reasoning']

    def test_record_execution_truncates_long_input(self):
        state = CollabState(query='Test')
        long_input = 'x' * 500
        state.record_execution('Agent1', long_input, 'output')

        step = state.execution_history[0]
        assert len(step['input']) <= 203  # 200 + '...'
        assert step['input'].endswith('...')

    def test_record_execution_truncates_long_output(self):
        state = CollabState(query='Test')
        long_output = 'y' * 500
        state.record_execution('Agent1', 'input', long_output)

        step = state.execution_history[0]
        assert len(step['output']) <= 203


class TestCollabRunResult:
    """Tests for CollabRunResult dataclass."""

    def test_basic_result(self):
        usage = RunUsage()
        result = CollabRunResult(output='Final output', usage=usage, iterations=3)

        assert result.output == 'Final output'
        assert result.iterations == 3
        assert str(result) == 'Final output'

    def test_execution_path_from_state(self):
        state = CollabState(query='Test')
        state.execution_path = ['Agent1', 'Agent2', 'Agent3']

        result = CollabRunResult(
            output='Done', usage=RunUsage(), _state=state
        )
        assert result.execution_path == ['Agent1', 'Agent2', 'Agent3']

    def test_execution_history_from_state(self):
        state = CollabState(query='Test')
        state.record_execution('Agent1', 'input', 'output')

        result = CollabRunResult(
            output='Done', usage=RunUsage(), _state=state
        )
        assert len(result.execution_history) == 1

    def test_all_messages_from_state(self):
        state = CollabState(query='Test')
        summary = AgentRunSummary(agent_name='Agent1', output='test')
        state.messages.append(summary)

        result = CollabRunResult(
            output='Done', usage=RunUsage(), _state=state
        )
        assert len(result.all_messages()) == 1

    def test_print_execution_flow(self):
        state = CollabState(query='Test')
        state.execution_path = ['Agent1', 'Agent2']
        state.record_execution('Agent1', 'query1', 'processing...')

        # Simulate handoff
        class MockHandoff:
            next_agent = 'Agent2'
            query = 'continue'
            reasoning = 'done with first step'

        state.record_execution('Agent1', 'query1', MockHandoff())
        state.record_execution('Agent2', 'continue', 'final answer')

        result = CollabRunResult(
            output='final answer',
            usage=RunUsage(),
            iterations=2,
            final_agent='Agent2',
            _state=state,
        )

        flow = result.print_execution_flow()
        assert 'Execution Flow' in flow
        assert 'Agent1' in flow
        assert 'Agent2' in flow
        assert 'Total iterations: 2' in flow
        assert 'Final agent: Agent2' in flow

    def test_max_iterations_reached_warning(self):
        state = CollabState(query='Test')
        result = CollabRunResult(
            output='partial',
            usage=RunUsage(),
            iterations=10,
            max_iterations_reached=True,
            _state=state,
        )

        flow = result.print_execution_flow()
        assert 'Max iterations reached' in flow


class TestAgentContext:
    """Tests for AgentContext model."""

    def test_default_values(self):
        ctx = AgentContext()
        assert ctx.messages == []
        assert ctx.last_output is None

    def test_with_values(self):
        ctx = AgentContext(messages=['msg1', 'msg2'], last_output='result')
        assert len(ctx.messages) == 2
        assert ctx.last_output == 'result'


class TestHandoffData:
    """Tests for HandoffData dataclass."""

    def test_required_fields(self):
        data = HandoffData(
            caller_agent_name='Caller', callee_agent_name='Callee'
        )
        assert data.caller_agent_name == 'Caller'
        assert data.callee_agent_name == 'Callee'
        assert data.previous_handoff_str == ''
        assert data.message_history is None
        assert data.include_thinking is False
        assert data.include_tool_calls_with_callee is False

    def test_with_all_fields(self):
        data = HandoffData(
            previous_handoff_str='Previous context',
            caller_agent_name='Caller',
            callee_agent_name='Callee',
            message_history=['msg1', 'msg2'],
            include_thinking=True,
            include_tool_calls_with_callee=True,
        )
        assert data.previous_handoff_str == 'Previous context'
        assert len(data.message_history) == 2
        assert data.include_thinking is True


class TestPromptBuilderContext:
    """Tests for PromptBuilderContext dataclass."""

    @pytest.fixture
    def mock_agent(self):
        agent = Agent('test', name='TestAgent')
        return CollabAgent(agent=agent, description='Test')

    def test_minimal_context(self, mock_agent):
        ctx = PromptBuilderContext(
            agent=mock_agent,
            final_agent=True,
            can_handoff=False,
            handoff_agents=[],
            tool_agents=[],
            called_as_tool=False,
        )
        assert ctx.agent == mock_agent
        assert ctx.final_agent is True
        assert ctx.can_handoff is False
        assert ctx.ascii_topology is None
        assert ctx.can_do_parallel_agent_calls is True

    def test_full_context(self, mock_agent):
        other_agent = CollabAgent(
            agent=Agent('test', name='Other'), description='Other'
        )
        ctx = PromptBuilderContext(
            agent=mock_agent,
            final_agent=False,
            can_handoff=True,
            handoff_agents=[other_agent],
            tool_agents=[other_agent],
            called_as_tool=True,
            ascii_topology='A→B',
            can_do_parallel_agent_calls=False,
        )
        assert len(ctx.handoff_agents) == 1
        assert len(ctx.tool_agents) == 1
        assert ctx.ascii_topology == 'A→B'
        assert ctx.can_do_parallel_agent_calls is False


class TestAgentRunSummary:
    """Tests for AgentRunSummary dataclass."""

    def test_basic_summary(self):
        summary = AgentRunSummary(agent_name='TestAgent', output='result')
        assert summary.agent_name == 'TestAgent'
        assert summary.output == 'result'
        assert summary.start_time is None
        assert summary.run_time is None
        assert summary.usage is None

    def test_with_usage_copies(self):
        usage = RunUsage()
        usage.requests = 5
        summary = AgentRunSummary(agent_name='Test', usage=usage)

        # Modify original
        usage.requests = 10

        # Summary should have copied value (post_init copies)
        assert summary.usage.requests == 5

    def test_messages_default_empty(self):
        summary = AgentRunSummary(agent_name='Test')
        assert summary.messages == []
