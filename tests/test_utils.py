"""Unit tests for pydantic_collab._utils module.

Tests the utility functions used for message history processing,
context building, and prompt generation.
"""

import pytest
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

from pydantic_collab import CollabAgent
from pydantic_collab._types import HandoffData, PromptBuilderContext, ensure_tuple
from pydantic_collab._utils import (
    PART_TO_STR,
    default_build_agent_prompt,
    get_context,
    get_tool_calls,
    message_history_to_text,
)


class TestEnsureTuple:
    """Tests for ensure_tuple utility function."""

    def test_none_returns_none(self):
        assert ensure_tuple(None) is None

    def test_tuple_returns_same(self):
        t = (1, 2, 3)
        assert ensure_tuple(t) == t

    def test_list_converts_to_tuple(self):
        assert ensure_tuple([1, 2, 3]) == (1, 2, 3)

    def test_set_converts_to_tuple(self):
        result = ensure_tuple({1, 2, 3})
        assert isinstance(result, tuple)
        assert set(result) == {1, 2, 3}

    def test_frozenset_converts_to_tuple(self):
        result = ensure_tuple(frozenset([1, 2, 3]))
        assert isinstance(result, tuple)
        assert set(result) == {1, 2, 3}

    def test_single_value_wraps_in_tuple(self):
        assert ensure_tuple('hello') == ('hello',)
        assert ensure_tuple(42) == (42,)

    def test_empty_list_returns_empty_tuple(self):
        assert ensure_tuple([]) == ()

    def test_single_element_list(self):
        assert ensure_tuple(['one']) == ('one',)


class TestMessageHistoryToText:
    """Tests for message_history_to_text function."""

    def test_empty_history_returns_empty_string(self):
        # Empty list or list with only first element (which is skipped)
        assert message_history_to_text([]) == ''

    def test_single_message_is_skipped(self):
        # First message is always skipped
        history = [
            ModelRequest(parts=[UserPromptPart(content='Hello')])
        ]
        assert message_history_to_text(history) == ''

    def test_user_prompt_included(self):
        history = [
            ModelRequest(parts=[UserPromptPart(content='First')]),
            ModelRequest(parts=[UserPromptPart(content='Second message')]),
        ]
        result = message_history_to_text(history)
        assert 'User:' in result
        assert 'Second message' in result

    def test_system_prompt_ignored_by_default(self):
        history = [
            ModelRequest(parts=[UserPromptPart(content='First')]),
            ModelRequest(parts=[SystemPromptPart(content='System instructions')]),
        ]
        result = message_history_to_text(history, ignore_system=True)
        assert 'System instructions' not in result

    def test_system_prompt_included_when_not_ignored(self):
        history = [
            ModelRequest(parts=[UserPromptPart(content='First')]),
            ModelRequest(parts=[SystemPromptPart(content='System instructions')]),
        ]
        result = message_history_to_text(history, ignore_system=False)
        assert 'System:' in result
        assert 'System instructions' in result

    def test_thinking_part_excluded_by_default(self):
        history = [
            ModelRequest(parts=[UserPromptPart(content='First')]),
            ModelResponse(
                parts=[ThinkingPart(content='Internal thinking')],
                timestamp='2024-01-01T00:00:00Z',
            ),
        ]
        result = message_history_to_text(history, include_thinking=False)
        assert 'Internal thinking' not in result

    def test_thinking_part_included_when_requested(self):
        history = [
            ModelRequest(parts=[UserPromptPart(content='First')]),
            ModelResponse(
                parts=[ThinkingPart(content='Internal thinking')],
                timestamp='2024-01-01T00:00:00Z',
            ),
        ]
        result = message_history_to_text(history, include_thinking=True)
        assert 'Internal thinking' in result

    def test_text_part_in_response(self):
        history = [
            ModelRequest(parts=[UserPromptPart(content='First')]),
            ModelResponse(
                parts=[TextPart(content='Model response text')],
                timestamp='2024-01-01T00:00:00Z',
            ),
        ]
        result = message_history_to_text(history)
        assert 'Model Response' in result
        assert 'Model response text' in result

    def test_model_request_label(self):
        history = [
            ModelRequest(parts=[UserPromptPart(content='First')]),
            ModelRequest(parts=[UserPromptPart(content='Second')]),
        ]
        result = message_history_to_text(history)
        assert 'Model Request' in result

    def test_tool_call_part_without_content(self):
        history = [
            ModelRequest(parts=[UserPromptPart(content='First')]),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='my_tool',
                        args={'arg1': 'value'},
                        tool_call_id='tool_001',
                    )
                ],
                timestamp='2024-01-01T00:00:00Z',
            ),
        ]
        result = message_history_to_text(history)
        # Should handle parts without content attribute
        assert 'ToolCallPart' in result

    def test_multi_part_message(self):
        history = [
            ModelRequest(parts=[UserPromptPart(content='First')]),
            ModelRequest(
                parts=[
                    UserPromptPart(content='User says'),
                    SystemPromptPart(content='System says'),
                ]
            ),
        ]
        result = message_history_to_text(history, ignore_system=False)
        assert 'User says' in result
        assert 'System says' in result


class TestGetToolCalls:
    """Tests for get_tool_calls function."""

    def test_empty_history_returns_empty(self):
        assert get_tool_calls([], 'SomeAgent') == ''

    def test_no_matching_agent_calls(self):
        history = [
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='call_agent',
                        args={'agent_name': 'OtherAgent', 'input': 'query'},
                        tool_call_id='call_001',
                    )
                ],
                timestamp='2024-01-01T00:00:00Z',
            )
        ]
        result = get_tool_calls(history, 'TargetAgent')
        assert result == ''

    def test_finds_call_agent_for_target(self):
        history = [
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='call_agent',
                        args={'agent_name': 'TargetAgent', 'input': 'help me'},
                        tool_call_id='call_001',
                    )
                ],
                timestamp='2024-01-01T00:00:00Z',
            )
        ]
        result = get_tool_calls(history, 'TargetAgent')
        assert 'Called Agent TargetAgent' in result
        assert 'help me' in result

    def test_finds_tool_return_for_call(self):
        history = [
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='call_agent',
                        args={'agent_name': 'TargetAgent', 'input': 'help me'},
                        tool_call_id='call_001',
                    )
                ],
                timestamp='2024-01-01T00:00:00Z',
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='call_agent',
                        content='Here is the help',
                        tool_call_id='call_001',
                    )
                ]
            ),
        ]
        result = get_tool_calls(history, 'TargetAgent')
        assert 'Called Agent TargetAgent' in result
        assert 'Response is Here is the help' in result

    def test_ignores_non_call_agent_tools(self):
        history = [
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='other_tool',
                        args={'agent_name': 'TargetAgent', 'input': 'query'},
                        tool_call_id='call_001',
                    )
                ],
                timestamp='2024-01-01T00:00:00Z',
            )
        ]
        result = get_tool_calls(history, 'TargetAgent')
        assert result == ''

    def test_multiple_calls_to_same_agent(self):
        history = [
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='call_agent',
                        args={'agent_name': 'TargetAgent', 'input': 'first query'},
                        tool_call_id='call_001',
                    )
                ],
                timestamp='2024-01-01T00:00:00Z',
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='call_agent',
                        content='first response',
                        tool_call_id='call_001',
                    )
                ]
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='call_agent',
                        args={'agent_name': 'TargetAgent', 'input': 'second query'},
                        tool_call_id='call_002',
                    )
                ],
                timestamp='2024-01-01T00:00:01Z',
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='call_agent',
                        content='second response',
                        tool_call_id='call_002',
                    )
                ]
            ),
        ]
        result = get_tool_calls(history, 'TargetAgent')
        assert 'first query' in result
        assert 'second query' in result
        assert 'first response' in result
        assert 'second response' in result


class TestGetContext:
    """Tests for get_context function."""

    def test_empty_handoff_data_returns_empty(self):
        data = HandoffData(
            caller_agent_name='Agent1',
            callee_agent_name='Agent2',
        )
        result = get_context(data)
        assert result == ''

    def test_includes_previous_handoff_string(self):
        data = HandoffData(
            previous_handoff_str='Previous context from earlier handoff',
            caller_agent_name='Agent1',
            callee_agent_name='Agent2',
        )
        result = get_context(data)
        assert 'Previous context from earlier handoff' in result

    def test_includes_message_history(self):
        history = [
            ModelRequest(parts=[UserPromptPart(content='First')]),
            ModelResponse(
                parts=[TextPart(content='Response text')],
                timestamp='2024-01-01T00:00:00Z',
            ),
        ]
        data = HandoffData(
            caller_agent_name='Agent1',
            callee_agent_name='Agent2',
            message_history=history,
        )
        result = get_context(data)
        assert 'Data From Agent Agent1' in result
        assert '## Message History' in result

    def test_includes_tool_calls_when_requested(self):
        history = [
            ModelRequest(parts=[UserPromptPart(content='First')]),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='call_agent',
                        args={'agent_name': 'Agent2', 'input': 'help'},
                        tool_call_id='call_001',
                    )
                ],
                timestamp='2024-01-01T00:00:00Z',
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='call_agent',
                        content='assistance provided',
                        tool_call_id='call_001',
                    )
                ]
            ),
        ]
        data = HandoffData(
            caller_agent_name='Agent1',
            callee_agent_name='Agent2',
            message_history=history,
            include_tool_calls_with_callee=True,
        )
        result = get_context(data)
        assert '## Previous Tool Calls you had' in result

    def test_excludes_tool_calls_when_not_requested(self):
        history = [
            ModelRequest(parts=[UserPromptPart(content='First')]),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='call_agent',
                        args={'agent_name': 'Agent2', 'input': 'help'},
                        tool_call_id='call_001',
                    )
                ],
                timestamp='2024-01-01T00:00:00Z',
            ),
        ]
        data = HandoffData(
            caller_agent_name='Agent1',
            callee_agent_name='Agent2',
            message_history=history,
            include_tool_calls_with_callee=False,
        )
        result = get_context(data)
        assert '## Previous Tool Calls' not in result


class TestDefaultBuildAgentPrompt:
    """Tests for default_build_agent_prompt function."""

    @pytest.fixture
    def mock_agent(self):
        """Create a minimal mock agent for testing."""
        from pydantic_ai import Agent

        agent = Agent('test', name='TestAgent')
        return CollabAgent(agent=agent, description='Test agent description')

    @pytest.fixture
    def another_agent(self):
        """Create another mock agent for handoffs/calls."""
        from pydantic_ai import Agent

        agent = Agent('test', name='OtherAgent')
        return CollabAgent(agent=agent, description='Other agent for testing')

    def test_no_handoffs_no_tools_minimal_output(self, mock_agent):
        ctx = PromptBuilderContext(
            agent=mock_agent,
            final_agent=True,
            can_handoff=False,
            handoff_agents=[],
            tool_agents=[],
            called_as_tool=False,
        )
        result = default_build_agent_prompt(ctx)
        # Should have final output instructions
        assert 'Final Output' in result or 'final_result' in result

    def test_with_handoff_agents(self, mock_agent, another_agent):
        ctx = PromptBuilderContext(
            agent=mock_agent,
            final_agent=False,
            can_handoff=True,
            handoff_agents=[another_agent],
            tool_agents=[],
            called_as_tool=False,
        )
        result = default_build_agent_prompt(ctx)
        assert 'HANDOFF' in result
        assert 'OtherAgent' in result
        assert 'Other agent for testing' in result

    def test_with_tool_agents(self, mock_agent, another_agent):
        ctx = PromptBuilderContext(
            agent=mock_agent,
            final_agent=True,
            can_handoff=False,
            handoff_agents=[],
            tool_agents=[another_agent],
            called_as_tool=False,
        )
        result = default_build_agent_prompt(ctx)
        assert 'CALL' in result or 'call_agent' in result
        assert 'OtherAgent' in result

    def test_both_handoffs_and_tools(self, mock_agent, another_agent):
        ctx = PromptBuilderContext(
            agent=mock_agent,
            final_agent=False,
            can_handoff=True,
            handoff_agents=[another_agent],
            tool_agents=[another_agent],
            called_as_tool=False,
        )
        result = default_build_agent_prompt(ctx)
        assert 'HANDOFF' in result
        assert 'TOOL' in result or 'CALL' in result

    def test_called_as_tool_context(self, mock_agent):
        ctx = PromptBuilderContext(
            agent=mock_agent,
            final_agent=True,
            can_handoff=False,
            handoff_agents=[],
            tool_agents=[],
            called_as_tool=True,
        )
        result = default_build_agent_prompt(ctx)
        assert 'tool' in result.lower()

    def test_includes_topology_when_provided(self, mock_agent, another_agent):
        ctx = PromptBuilderContext(
            agent=mock_agent,
            final_agent=False,
            can_handoff=True,
            handoff_agents=[another_agent],
            tool_agents=[],
            called_as_tool=False,
            ascii_topology='Agents: A, B\nStart: A, Final: B\nHandoffs: A→B',
        )
        result = default_build_agent_prompt(ctx)
        assert 'Topology' in result
        assert 'A→B' in result

    def test_parallel_calls_enabled(self, mock_agent, another_agent):
        ctx = PromptBuilderContext(
            agent=mock_agent,
            final_agent=True,
            can_handoff=False,
            handoff_agents=[],
            tool_agents=[another_agent],
            called_as_tool=False,
            can_do_parallel_agent_calls=True,
        )
        result = default_build_agent_prompt(ctx)
        assert 'parallel' in result.lower()

    def test_parallel_calls_disabled(self, mock_agent, another_agent):
        ctx = PromptBuilderContext(
            agent=mock_agent,
            final_agent=True,
            can_handoff=False,
            handoff_agents=[],
            tool_agents=[another_agent],
            called_as_tool=False,
            can_do_parallel_agent_calls=False,
        )
        result = default_build_agent_prompt(ctx)
        assert 'sequentially' in result.lower() or 'not run in parallel' in result.lower()


class TestPartToStrMapping:
    """Tests for PART_TO_STR constant mapping."""

    def test_user_prompt_maps_to_user(self):
        assert PART_TO_STR[UserPromptPart] == 'User:'

    def test_system_prompt_maps_to_system(self):
        assert PART_TO_STR[SystemPromptPart] == 'System:'

    def test_text_part_maps_to_empty(self):
        assert PART_TO_STR[TextPart] == ''
