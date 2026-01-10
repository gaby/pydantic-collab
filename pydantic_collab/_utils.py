"""Internal utility functions for agent Collabs.

This module contains helper functions for building prompts, processing message
history, and extracting tool calls. Users should not import from this module directly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from pydantic_ai import (
    BaseToolCallPart,
    ModelRequest,
    SystemPromptPart,
    TextPart,
    ThinkingPart,
    ToolReturnPart,
    UserPromptPart,
)

if TYPE_CHECKING:
    from ._types import HandoffData, PromptBuilderContext

# =============================================================================
# Message History Utilities
# =============================================================================


PART_TO_STR: dict[Any, str] = {UserPromptPart: 'User:', SystemPromptPart: 'System:', TextPart: ''}


def message_history_to_text(
    mh: list[Any],
    ignore_system: bool = True,
    include_thinking: bool = False,
    include_instructions: bool = False,
) -> str:
    """Convert message history to human-readable text.

    Args:
        mh: List of model requests and responses
        ignore_system: Whether to skip system prompt parts
        include_thinking: Whether to include thinking parts
        include_instructions: Whether to include instruction parts (pydantic_ai feature)

    Returns:
        Formatted text representation of message history
    """
    lines: list[str] = []
    # We skip the first, it's important because that's the initialization!!
    for mr in mh[1:]:
        lines.append(str('Model Request' if isinstance(mr, ModelRequest) else 'Model Response'))
        for part in getattr(mr, 'parts', []):
            if ignore_system and isinstance(part, SystemPromptPart):
                continue
            if not include_thinking and isinstance(part, ThinkingPart):
                continue
            # Handle parts without content attribute (like ToolCallPart, ToolReturnPart)
            content = getattr(part, 'content', None)
            if content is not None:
                lines.append(str(f'{PART_TO_STR.get(type(part), part.__class__.__name__)} {content}'))
            else:
                lines.append(str(f'{PART_TO_STR.get(type(part), part.__class__.__name__)}'))
        if include_instructions and (inst := getattr(mr, 'instructions', None)):
            # instructions can be a sequence of mixed types (str or callables). Ensure
            # everything is stringified before joining to avoid unknown element types.
            lines.append(f'\tinstructions: {chr(10).join(map(str, inst))}')
    return '\n'.join(lines)


def get_tool_calls(mh: list[Any], agent_name: str) -> str:
    """Extract tool calls to a specific agent from message history.

    Finds all call_agent tool calls targeting the specified agent and their returns.

    Args:
        mh: List of model requests and responses
        agent_name: Name of the agent to find calls for

    Returns:
        Formatted text of tool calls and responses
    """
    lines: list[str] = []
    call_ids: set[str] = set()
    for mg in mh:
        for part in getattr(mg, 'parts', []):
            if isinstance(part, BaseToolCallPart) and isinstance(getattr(part, 'args', None), dict):
                args = cast(dict[str, Any], getattr(part, 'args'))
                if getattr(part, 'tool_name', None) == 'call_agent' and args.get('agent_name') == agent_name:
                    inp = args.get('input')
                    lines.append(str(f'Called Agent {agent_name} with query {inp}'))
                    call_ids.add(str(getattr(part, 'tool_call_id', '')))
            if isinstance(part, ToolReturnPart) and getattr(part, 'tool_call_id', None) in call_ids:
                resp = getattr(part, 'content', None)
                lines.append(str(f'Response is {resp}'))
    return '\n'.join(lines)


# =============================================================================
# Context Builders
# =============================================================================


def get_context(handoff_data: HandoffData) -> str:
    """Build context string from handoff data.

    Default context builder that assembles context from previous handoffs,
    message history, and tool calls.

    Args:
        handoff_data: Data about the handoff including history and settings

    Returns:
        Formatted context string to inject into next agent's instructions
    """
    out: list[str] = []
    if handoff_data.previous_handoff_str:
        out.append(handoff_data.previous_handoff_str)
    if handoff_data.message_history:
        out.append(f'Data From Agent {handoff_data.caller_agent_name}')
        out.append(
            '## Message History\n'
            + message_history_to_text(
                handoff_data.message_history,
                include_thinking=handoff_data.include_thinking,
            )
        )
        if handoff_data.include_tool_calls_with_callee:
            tc = get_tool_calls(handoff_data.message_history, handoff_data.callee_agent_name)
            if tc:
                out.extend(['## Previous Tool Calls you had', tc])
    return '\n'.join(out)


# =============================================================================
# Prompt Builders
# =============================================================================


def default_build_agent_prompt(ctx: PromptBuilderContext) -> str:
    """Build agent-specific instructions based on capabilities.

    Default prompt builder that describes available actions (handoffs, tool calls,
    final output) based on the agent's position in the Collab topology.

    Args:
        ctx: Context with agent capabilities and available targets

    Returns:
        Formatted instructions string
    """
    output_instructions: list[str] = []
    # Add general best practices for all agents in Collabs

    # Header explaining multi-agent capabilities
    has_handoffs = ctx.can_handoff and ctx.handoff_agents
    has_tool_agents = bool(ctx.tool_agents)

    if has_handoffs or has_tool_agents:
        output_instructions.extend(
            ['## Tool Usage Best Practices', '', '⚠️ **CRITICAL GUIDELINES** for using tools efficiently:\n', '']
        )

        output_instructions.append('## Multi-Agent Collaboration')
        output_instructions.append('')

        if has_handoffs and has_tool_agents:
            output_instructions.append('You have TWO DISTINCT ways to interact with other agents:\n')
            output_instructions.append(
                '### 1. HANDOFF (Transfer Control)\n'
                '- **Purpose**: Permanently transfer control when your part is done\n'
                "- **Mechanism**: Use HandoffOutput with 'next_agent' field\n"
                '- **Result**: The other agent takes over completely; you stop processing\n'
                '- **Use when**: Another agent should handle the rest of the conversation\n'
                "- **⚠️ Important**: If control is handed back to you later, you won't have your previous conversation history\n"
            )
            output_instructions.append(
                '### 2. TOOL CALL (Get Help, Stay in Control)\n'
                '- **Purpose**: Get specific information while remaining in control\n'
                '- **Mechanism**: Use the call_agent tool function\n'
                '- **Result**: You receive their response and continue your work\n'
                '- **Use when**: You need help but will continue processing afterward\n'
            )
            output_instructions.append(
                '⚠️ **IMPORTANT**: Agents available for HANDOFF are DIFFERENT from agents available as TOOLS. '
                'Check the sections below carefully!\n'
            )
        elif has_handoffs:
            output_instructions.append(
                'You can **HANDOFF** control to other agents. This permanently transfers control - '
                'the receiving agent will continue from that point and you will stop processing.\n'
                "Use HandoffOutput with 'next_agent' to specify the target agent.\n\n"
                "⚠️ **Important**: If control is handed back to you later, you won't have your previous conversation history.\n"
            )
        elif has_tool_agents:
            output_instructions.append(
                'You can **CALL** other agents as tools to get specific information. '
                'Use the call_agent tool function to request help. '
                'After they respond, you continue your work with their output.\n'
            )

    # Handoff instructions
    if has_handoffs:
        agents_desc: list[str] = []
        for agent in ctx.handoff_agents:
            agents_desc.append(f'  • **{agent.name}**: {agent.description}')

        output_instructions.append('---')
        output_instructions.append('## Handoff')
        output_instructions.append(f'You are {ctx.agent.name}.')  # We need this for the agent to understand where it
        # Stands in relation to topology and other things
        output_instructions.append(
            '**Hand off promptly**: When your task is complete, hand off immediately.\n'
            'Do not linger or double-check endlessly.\n'
        )
        output_instructions.append('### Agents You Can HANDOFF To')
        output_instructions.append(
            "These agents can take over complete control. Use HandoffOutput(next_agent='AgentName', query='...'):\n"
        )
        output_instructions.append('\n'.join(agents_desc))
        if ctx.ascii_topology:
            output_instructions.extend(['### Topology of Agent Handoff Connections', ctx.ascii_topology])
        output_instructions.append('---')
        output_instructions.append('')

    # Tool agent instructions
    if has_tool_agents:
        agents_desc: list[str] = []
        for agent in ctx.tool_agents:
            agents_desc.append(f'  • **{agent.name}**: {agent.description}')

        output_instructions.append('---')
        output_instructions.append('### Agents You Can CALL as Tools')
        output_instructions.append(
            "These agents can be called as functions. Use call_agent(agent_name='AgentName', input='...'):\n"
        )
        if ctx.can_do_parallel_agent_calls:
            output_instructions.append(
                'You may call the Agents in parallel to save time. Do that when calling different agents, '
                'best not to use the same agent in parallel'
            )
        else:
            output_instructions.append(
                'Agents do not run in parallel, so if you call more than one agent at a time, they '
                'will run sequentially and will not save time'
            )
        output_instructions.append('\n'.join(agents_desc))
        output_instructions.append('')
        output_instructions.append(
            'Note: After calling an agent as a tool, you remain in control and can continue processing.'
        )
        output_instructions.append('')

    # Final output instructions
    if ctx.final_agent:
        output_instructions.append('---')
        output_instructions.append('### Returning Final Output')
        if ctx.called_as_tool:
            # TODO: Maybe change that
            output_instructions.append('You have been called as a tool by another Agent')
        output_instructions.append(
            "When your task is complete and you're ready to respond to the user, call final_result"
        )
        output_instructions.append('')

    output_str = 's\n'.join(output_instructions)

    return output_str
