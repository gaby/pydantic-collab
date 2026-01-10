"""Internal types, settings, and data structures for agent Collabs.

This module contains all type definitions, settings classes, and data structures
used internally by the Collab orchestrator. Users should not import from this module directly.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, ClassVar, Generic, Literal, TypeAlias, TypeVar, cast

from pydantic import BaseModel, Field
from pydantic_ai import RunUsage
from pydantic_ai.agent import AbstractAgent
from typing_extensions import TypeAliasType

T = TypeVar('T')
# =============================================================================
# Utility Functions (defined here to avoid circular imports)
# =============================================================================


# For some reason tuple[Unpack[T]]  doesn't work in ruff
def ensure_tuple(value: T) -> tuple[T] | T | None:
    """Convert a value to a tuple, handling various input types.

    Args:
        value: Value to convert - can be None, tuple, list, set, frozenset, or single item

    Returns:
        None if value is None, the original tuple if already a tuple,
        a tuple of the items if a collection, or a single-item tuple for other values
    """
    if value is None:
        return None
    if isinstance(value, tuple):
        return value
    elif isinstance(value, (list, set, frozenset)):
        return tuple(value)
    return (value,)


# =============================================================================
# Type Aliases
# =============================================================================
RegularOptions: TypeAlias = Literal['force', 'disallow', 'allow']
t_agent_name: TypeAlias = str
T = TypeVar('T')
AgentDepsT = TypeVar('AgentDepsT')
OutputDataT = TypeVar('OutputDataT')
# Can be an agent name, an AbstractAgent instance, or a CollabAgent wrapper
t_agent_desc: TypeAlias = 'str | AbstractAgent | CollabAgent'
t_seq_or_one = TypeAliasType('t_seq_or_one', T | Sequence[T], type_params=(T,))

# =============================================================================
# Exceptions
# =============================================================================


class CollabError(Exception):
    """Raised when Collab configuration is invalid."""

    pass


# =============================================================================
# Agent Context
# =============================================================================


class AgentContext(BaseModel):
    """Context data for a single agent."""

    messages: list[Any] = Field(default_factory=lambda: cast(list[Any], []))
    last_output: str | None = None


@dataclass
class AgentRunSummary:
    agent_name: str
    start_time: datetime | None = None
    run_time: float | int | None = None
    output: Any = None
    usage: RunUsage | None = None
    messages: list[Any] = field(default_factory=lambda: cast(list[Any], []))

    def __post_init__(self):
        # This is done because usage gets up and up and it's a mutable object. This
        # is not a 100% valid method as it
        # could climb between __init__ and __post_init__,
        # especially with Threads, but it's close enough
        if self.usage is not None:
            self.usage = copy.copy(self.usage)


@dataclass
class CollabState:
    """Dynamic state for a Collab execution.

    Automatically creates per-agent contexts based on registered agents.
    """

    query: str
    final_output: str = ''
    agent_contexts: dict[str, AgentContext] = field(default_factory=lambda: cast(dict[str, AgentContext], {}))
    execution_path: list[str] = field(default_factory=lambda: cast(list[str], []))
    execution_history: list[dict[str, Any]] = field(default_factory=lambda: cast(list[dict[str, Any]], []))
    messages: list[AgentRunSummary] = field(default_factory=lambda: cast(list[AgentRunSummary], []))

    def get_context(self, agent_name: str) -> AgentContext:
        """Get or create context for an agent."""
        if agent_name not in self.agent_contexts:
            self.agent_contexts[agent_name] = AgentContext()
        return self.agent_contexts[agent_name]

    def record_execution(self, agent_name: str, input_query: str, output: Any):
        """Record an agent execution in history."""
        # Extract output content. Avoid referencing HandOffBase type
        # (defined later) to keep this function lightweight for the type
        # checker — detect a handoff-like object by required attributes.
        if hasattr(output, 'next_agent') and hasattr(output, 'query'):
            output_text = getattr(output, 'query')
            next_agent = getattr(output, 'next_agent', None)
            action = 'handoff'
            reasoning = getattr(output, 'reasoning', '') or ''
        else:
            output_text = str(output)
            next_agent = None
            action = 'final'
            reasoning = ''
        self.execution_history.append(
            {
                'agent': agent_name,
                'action': action,
                'input': input_query[:200] + '...' if len(input_query) > 200 else input_query,
                'output': str(output_text)[:200] + '...' if len(str(output_text)) > 200 else str(output_text),
                'next_agent': next_agent,
                'reasoning': reasoning[:100] + '...' if len(reasoning) > 100 else reasoning,
            }
        )


@dataclass
class CollabRunResult(Generic[OutputDataT]):
    """The final result of an agent run."""

    output: OutputDataT
    """The output data from the agent run."""
    usage: RunUsage
    """Total usage of all agents in this run"""
    iterations: int = 0
    """Number of agent handoffs executed. does not include agent tool call"""
    final_agent: str = ''
    """Name of the agent that produced final output. Right now can be foreseeable,
     in the future possibly not"""
    max_iterations_reached: bool = False
    """Whether the iteration limit was hit and it was stopped unplanned"""

    _state: CollabState = field(repr=False, compare=False, default_factory=lambda: CollabState(query=''))

    def all_messages(self) -> list[AgentRunSummary]:
        """Return all messages."""
        return self._state.messages

    @property
    def execution_path(self) -> list[str]:
        """Get the path of agents that were executed."""
        return self._state.execution_path

    @property
    def execution_history(self) -> list[dict[str, Any]]:
        """Get the full execution history with inputs/outputs."""
        return self._state.execution_history

    def print_execution_flow(self) -> str:
        """Generate a readable execution flow diagram."""
        lines = [
            'Execution Flow:',
            '=' * 50,
        ]

        for i, step in enumerate(self.execution_history):
            prefix = '→ ' if i > 0 else '⚡ '
            agent = step['agent']
            action = step['action']

            lines.append(f'{prefix}{agent}')
            lines.append(f'   Input: {step["input"][:80]}...')
            lines.append(f'   Output: {step["output"][:80]}...')
            if step['reasoning']:
                lines.append(f'   Reason: {step["reasoning"][:60]}...')

            if action == 'handoff':
                lines.append(f'   ↓ routing to {step.get("next_agent", "unknown")}')
            else:
                lines.append('   ✓ FINAL OUTPUT')
            lines.append('')

        lines.append(f'Total iterations: {self.iterations}')
        lines.append(f'Final agent: {self.final_agent}')
        if self.max_iterations_reached:
            lines.append('⚠️  Max iterations reached!')

        return '\n'.join(lines)

    def __str__(self) -> str:
        return str(self.output)


# =============================================================================
# Agent Connection/Wrapper
# =============================================================================


@dataclass(frozen=True)
class CollabAgent:
    """Wrapper for an agent participating in a Collab.

    Defines an agent's role in the Collab including which agents it can
    call as tools and which agents it can hand off to.
    """

    agent: AbstractAgent
    """The underlying pydantic-ai Agent."""

    description: str | None = None
    """Human-readable description of this agent's purpose."""

    agent_calls: Sequence[t_agent_desc] = ()
    """Other agents this agent can call as tools
     (synchronous calls that return to the caller agent).
    Can be agent names (str), AbstractAgent instances, or CollabAgent instances."""

    agent_handoffs: Sequence[t_agent_desc] = ()
    """Other agents this agent can hand off to (transfer control).
    Can be agent names (str), AbstractAgent instances, or CollabAgent instances."""
    name: str | None = None
    """Name of this Collab Agent. Used by other agents for choosing how to hand over"""

    def __init__(
        self,
        agent: AbstractAgent,
        description: str | None,
        agent_calls: t_agent_desc | Sequence[t_agent_desc] = (),
        agent_handoffs: t_agent_desc | Sequence[t_agent_desc] = (),
        name: str | None = None,
    ) -> None:
        """Initialize a CollabAgent.

        Args:
            agent: The underlying pydantic-ai Agent.
            description: LLM readable description of this agent. Required for all agents that aren't the starting agent.
            agent_calls: Which other agents can be called
            agent_handoffs: Which other agents can hand off to
            name: Name of the agent, if not specified in the agent itself

        """
        object.__setattr__(self, 'agent', agent)
        object.__setattr__(self, 'description', description)
        object.__setattr__(self, 'agent_calls', ensure_tuple(agent_calls))
        object.__setattr__(self, 'agent_handoffs', ensure_tuple(agent_handoffs))
        if name is None and agent.name is None:
            raise ValueError('Agent must have a name to participate in Collab')
        if name is None:
            name = agent.name
        object.__setattr__(self, 'name', name)

    def __hash__(self) -> int:
        """Make CollabAgent hashable based on agent name."""
        return hash(hash(self.name) ^ hash(self.description))

    def __eq__(self, other: Any) -> bool:
        """Compare CollabAgent instances by agent name."""
        if isinstance(other, CollabAgent):
            return (
                self.name == other.name
                # self.agent == other.agent #and
                # self.description == other.description
            )
        return False

    def __repr__(self) -> str:
        """String representation showing name and description."""
        return f'CollabAgent(name={self.name!r}, description={self.description!r})'

    @property
    def requires_deps(self) -> bool:
        return self.agent.deps_type is not type(None)


# =============================================================================
# Output Types
# =============================================================================
class HandOffBase(BaseModel, Generic[T]):
    """Route to another agent for further processing.

    Use this when you finished with your work and want to hand off.
    If you just need help, call tools
    Example:
        HandoffOutput(
            next_agent="Expert",
            query="Please analyze this specific aspect..."
        )
    """

    reasoning: str | None = Field(
        default=None,
        description='Optional explanation of why this action was taken. Useful for debugging and transparency.',
    )
    next_agent: str = Field(description='Name of the agent to route to.')
    query: T = Field(description='The query/message/response to send to the next agent.')


# =============================================================================
# Context Data Structures
# =============================================================================


@dataclass
class PromptBuilderContext:
    """Context passed to prompt builder functions.

    Users can create custom prompt builders that accept this context.
    """

    agent: CollabAgent
    final_agent: bool
    can_handoff: bool
    handoff_agents: Sequence[CollabAgent]
    tool_agents: Sequence[CollabAgent]
    called_as_tool: bool
    ascii_topology: str | None = None
    can_do_parallel_agent_calls: bool = True


@dataclass(repr=False, kw_only=True)
class HandoffData:
    """Data passed during agent handoffs."""

    previous_handoff_str: str = ''
    caller_agent_name: str
    callee_agent_name: str
    message_history: list[Any] | None = None
    include_thinking: bool = False
    include_tool_calls_with_callee: bool = False


# =============================================================================
# Settings
# =============================================================================


@dataclass
class CollabSettings:
    """Network behavior configuration for agent Collabs.

    Controls how data flows between agents during handoffs and what information
    is included in context.
    """

    include_thinking: RegularOptions = 'disallow'
    """Control whether thinking parts are included in handoff context.
    - "allow": Agent decides via HandoffOutput.include_thinking
    - "force": Always include (field becomes ClassVar[bool] = True)
    - "disallow": Never include (field becomes ClassVar[bool] = False)
    """

    include_conversation: RegularOptions = 'allow'
    """Control whether conversation history is included in handoff context.
    - "allow": Agent decides via HandoffOutput.include_conversation
    - "force": Always include
    - "disallow": Never include
    """

    include_handoff: RegularOptions = 'allow'
    """Control whether previous handoff context is accumulated.
    - "allow": Agent decides via HandoffOutput.include_previous_handoff
    - "force": Always include
    - "disallow": Never include
    """

    include_tool_calls_with_callee: RegularOptions = 'allow'
    """Control whether tool calls with target agent are included.
    - "allow": Agent decides via HandoffOutput.include_tool_calls_with_callee
    - "force": Always include
    - "disallow": Never include
    """

    output_restrictions: Literal['only_str', 'only_original', 'str_or_original'] = 'str_or_original'
    """Control what types agents can output in their query/response fields.
    - "only_str": Agents can only output strings
    - "only_original": Agents must use their configured _output_type
    - "str_or_original": Agents can output either string or their _output_type
    """

    prompt_builder: Callable[[PromptBuilderContext], str] | None = None
    """Custom function to build agent-specific instructions.
    If None, uses default_build_agent_prompt from _utils module.
    """

    context_builder: Callable[[HandoffData], str] | None = None
    """Custom function to build context string from handoff data.
    If None, uses get_context from _utils module.
    """

    include_topology_in_prompt: bool = True
    """If true, will include topology of the handoffs available between agents, 
    if handoffs are available"""


def get_right_handoff_model(settings: CollabSettings) -> type[HandOffBase[Any]]:
    """Dynamically create HandoffOutput class based on Collab settings.

    Converts "force"/"disallow" settings into ClassVar fields that enforce
    specific values rather than allowing agent control.

    Args:
        settings: The Collab network settings to apply

    Returns:
        A HandOffBase subclass with appropriate field types
    """
    static = ('force', 'disallow')

    class HandOffOutput(HandOffBase[T]):
        if settings.include_conversation in static:
            include_conversation: ClassVar[bool] = settings.include_conversation == 'force'
        else:
            include_conversation: bool = Field(
                default=False,
                description='Whether to include the full conversation history. '
                'Useful for complex multi-step reasoning.',
            )

        if settings.include_handoff in static:
            include_previous_handoff: ClassVar[bool] = settings.include_handoff == 'force'
        else:
            include_previous_handoff: bool = Field(
                default=False, description='Whether to include the previous handoff history.'
            )

        if settings.include_tool_calls_with_callee in static:
            include_tool_calls_with_callee: ClassVar[bool] = settings.include_tool_calls_with_callee == 'force'
        else:
            include_tool_calls_with_callee: bool = Field(
                default=False,
                description='Whether to include tool calls with that agent if you had any previous interactions.',
            )

        if settings.include_thinking in static:
            include_thinking: ClassVar[bool] = settings.include_thinking == 'force'
        else:
            include_thinking: bool = Field(
                default=False, description='Whether to include thinking parts from the conversation history.'
            )

    return HandOffOutput


# Used to redeclare it as is
# t_agent_desc already declared above.
