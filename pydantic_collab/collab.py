import datetime
import time
from asyncio import Lock
from collections import defaultdict
from collections.abc import Callable, Collection, Sequence
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Literal,
    Self,
    cast,
    overload,
)

from pydantic.json_schema import GenerateJsonSchema

# =============================================================================
# Result Types
# =============================================================================
from pydantic_ai import (
    AbstractToolset,
    AgentRunResult,
    FunctionToolset,
    ModelRequest,
    ModelResponse,
    ModelRetry,
    ModelSettings,
    RunContext,
    RunUsage,
    Tool,
    ToolsetFunc,
    UsageLimits,
)
from pydantic_ai.agent.abstract import AbstractAgent, Instructions
from pydantic_ai.models import KnownModelName, Model
from pydantic_ai.tools import (
    DocstringFormat,
    GenerateToolJsonSchema,
    ToolFuncContext,
    ToolFuncEither,
    ToolFuncPlain,
    ToolParams,
    ToolPrepareFunc,
)

# SystemPromptFunc not used directly here; avoid unused import
from ._types import (
    AgentDepsT,
    AgentRunSummary,
    CollabAgent,
    CollabError,
    CollabRunResult,
    CollabSettings,
    CollabState,
    HandOffBase,
    HandoffData,
    OutputDataT,
    PromptBuilderContext,
    ensure_tuple,
    get_right_handoff_model,
    t_agent_desc,
    t_agent_name,
    t_seq_or_one,
)
from ._utils import default_build_agent_prompt, get_context

if TYPE_CHECKING:
    try:
        import logfire
    except ImportError:
        logfire = None
    try:
        import matplotlib
    except ImportError:
        matplotlib = None


@dataclass
class Collab(Generic[AgentDepsT, OutputDataT]):
    """Orchestrates a multi-agent system from declarative connections.

    Usage:
        agent_a = Agent('openai:gpt-4o-mini', name='AgentA', ...)
        agent_b = Agent('openai:gpt-4o-mini', name='AgentB', ...)
        agent_c = Agent('openai:gpt-4o-mini', name='AgentC', ...)

        conn1 = AgentConnection(agent_a, (agent_b, agent_c), connection_type="bidirectional")
        conn2 = AgentConnection(agent_c, agent_d, connection_type="forward")

        response = Collab(
            starting_agent=agent_a,
            final_agents=(agent_a, agent_b),
            connections=(conn1, conn2)
        ).run("Hello!")
    """

    agents: Sequence[CollabAgent | tuple[AbstractAgent, str | None]]
    """
    All agents participating in the Collab, Can be either:
    - CollabAgent
    - tuple[pydantic_ai.Agent, str | None] - pydantic ai agent and description. Relevant   
    
    
    
    """
    starting_agent: CollabAgent | None = None
    """Entry point agent; defaults to first agent in agents when no set."""
    final_agent: CollabAgent | None = None
    """Agent that will produce final output; defaults to starting_agent if not set, unless a custom agent Collab is used"""
    name: str | None = None
    """Optional name for the Collab, used for logging."""
    max_handoffs: int = 10
    """Maximum number of agent executions to prevent infinite loops"""
    tools: Sequence[Tool | ToolFuncEither[AgentDepsT, ...]] | None = None
    """Function tools available to all agents in the Collab"""
    model: Model | KnownModelName | str | None = None
    """When set, would use this model for all the agents and override their model. Otherwise the agents' models will be used."""
    model_settings: ModelSettings | None = None
    """Model configuration settings (temperature, max_tokens, etc.)"""
    usage_limits: UsageLimits = field(default_factory=UsageLimits)
    """Token and request usage limits"""
    instructions: Instructions[AgentDepsT] = None
    """Dynamic instructions passed to agents"""
    """Auto-build topology: star (hub), mesh (full), or forward_handoff (chain)"""
    max_agent_call_depth: int = 3

    # Internal state

    _instrument_logfire: bool = field(init=False, default=True)
    _agent_tools: dict[CollabAgent, tuple[Any, ...]] = field(
        init=False, default_factory=lambda: defaultdict(tuple), repr=False
    )
    _agents: tuple[CollabAgent, ...] = field(default_factory=tuple, init=False, repr=False)

    _collab_settings: CollabSettings = field(init=False, default_factory=CollabSettings)
    """Collab behavior configuration (handoff options, prompt builder, etc.)"""

    _toolsets: tuple[AbstractToolset, ...] = field(default_factory=tuple, init=False, repr=False)
    _user_toolset: FunctionToolset = field(init=False, repr=False)
    _dynamic_toolsets: list[Any] = field(init=False, default_factory=list, repr=False)
    _toolset_by_agent: dict[CollabAgent, Any] = field(init=False, default_factory=dict, repr=False)
    _handoffs: dict[CollabAgent, tuple[CollabAgent, ...]] = field(init=False, default_factory=dict, repr=False)
    _name_to_agent: dict[str, CollabAgent] = field(init=False, default_factory=dict[str, CollabAgent], repr=False)
    _instructions: Any = field(init=False, default_factory=tuple, repr=False)
    _logfire: 'logfire | None' = field(init=False, repr=False)
    # Context manager state
    _enter_lock: Lock = field(default_factory=Lock, init=False, repr=False)
    _entered_count: int = field(default=0, init=False, repr=False)
    _exit_stack: AsyncExitStack | None = field(default=None, init=False, repr=False)
    _handoff_model: type[HandOffBase[Any]] | None | None = field(default=None, init=False, repr=False)
    _allow_parallel_agent_calls: bool = field(default=True, init=False, repr=False)

    def __init__(
        self,
        agents: t_seq_or_one[CollabAgent | tuple[AbstractAgent, str | None]],
        starting_agent: CollabAgent | AbstractAgent | tuple[AbstractAgent, str | None] | None = None,
        final_agent: CollabAgent | str | None = None,
        name: str | None = None,
        max_handoffs: int = 10,
        output_type: OutputDataT | None = None,
        tools: t_seq_or_one[Tool | ToolFuncEither[AgentDepsT, ...]] | None = None,
        toolsets: t_seq_or_one[AbstractToolset[AgentDepsT]] | None = None,
        model: Model | KnownModelName | str | None = None,
        model_settings: ModelSettings | None = None,
        collab_settings: CollabSettings | None = None,
        usage_limits: UsageLimits | None = None,
        instructions: Instructions[AgentDepsT] = None,
        max_agent_call_depth: int = 3,
        allow_parallel_agent_calls: bool = True,
        allow_back_handoff: bool = True,
        instrument_logfire: bool = True,
    ) -> None:
        """Initiate a Collab.

        Args:
            agents: A list of agents with calls and handoffs specified. Agents that don't have calls can be specified
                as (pydantic_ai.Agentx, <description>), Agent that doesn't need description can be specified pydantic_ai.Agent
            starting_agent: Agent that will produce final output; defaults to starting_agent if not set, unless a custom
                agent Collab is used
            final_agent: Agent that will produce final output; defaults to starting_agent if not set, unless a custom
                agent Collab is used,
            name: Optional name for the Collab, used for logging
            max_handoffs: maximum number of handoffs allowed (independent of agent tool calls).
                0 means no handoffs. defaults to 0.
            output_type: Expected output type; overrides the final_agent's output type if specified,
            tools: tools to add to all agents, in addition to any tools already specified for the agents.
            toolsets: Toolsets to register with all the agents in the collab, including MCP servers and
                functions which take a run context

            model: When set, would use this model for all the agents and override their model.
                Otherwise, the agents' models will be used,
            model_settings: Model configuration settings. If set, will override Model settings set before on the agents.
            collab_settings: Settings for how to handle handoffs between agents,
            usage_limits: Usage limits that are applicable to the entire run. when None, default UsageLimits are used
            instructions: Instructions[AgentDepsT] = additional instructions used for all agents.
                Doesn't override any other instructions specified for the agents
            max_agent_call_depth: Maximum depth of agent tool calls allowed. i.e. setting it to 1 will allow
                Agent A calling Agent B as tool, will prevent Agent A calling Agent B calling Agent C. Doesn't relate to
                handoffs. Defaults to 3
            allow_parallel_agent_calls: Whether to allow parallel agent calls. Defaults to True
            allow_back_handoff: Whether to allow an Agent to return work to an Agent that has already handed off before
                (Right before or any other time before)
            instrument_logfire: Whether to instruments Logfire. If true, logfire will be used if it can be imported and
                has *already* been configured.
        """
        self._logger = None
        self._name_to_agent = {}
        self._instrument_logfire = instrument_logfire
        self._init_logfire()
        # Call post-init logic

        tmp_agents: list[CollabAgent] = []
        for agent in ensure_tuple(agents):
            tmp_agents.append(self._normalize_agent(agent, True))

        # Here we collect also agents that are listed by agent calls but aren't listed in another way
        c = 0
        agent_names = set(i.name for i in tmp_agents)
        while c < len(tmp_agents):
            for agent in (*tmp_agents[c].agent_calls, *tmp_agents[c].agent_handoffs):
                if isinstance(agent, CollabAgent) and agent.name not in agent_names:
                    tmp_agents.append(agent)
                    agent_names.add(agent.name)
            c += 1

        self._agents = tuple(tmp_agents)
        self._name_to_agent = {agent.name: agent for agent in self._agents}

        if starting_agent is not None:
            self.starting_agent = self._normalize_and_reuse_agent(starting_agent)

        if final_agent is not None:
            self.final_agent = self._normalize_and_reuse_agent(final_agent)

        # Validate agents names aren't identical.
        if isinstance(instructions, str):
            self._instructions = [instructions]
        elif instructions is not None:
            self._instructions = instructions
        else:
            self._instructions = []

        if tools:
            self._user_toolset = FunctionToolset(ensure_tuple(tools))
        else:
            self._user_toolset = FunctionToolset()
        self._toolsets = (self._user_toolset, *(ensure_tuple(toolsets) or ()))

        # Initialize public fields
        self.name = name
        self.max_handoffs = max_handoffs
        self._output_type = output_type
        self.model = model
        self.model_settings = model_settings
        self._collab_settings = collab_settings if collab_settings is not None else CollabSettings()
        self.usage_limits = usage_limits if usage_limits is not None else UsageLimits()
        self.max_agent_call_depth = max_agent_call_depth
        self._allow_parallel_agent_calls = allow_parallel_agent_calls

        # Initialize internal fields
        self._agent_tools = defaultdict(tuple)
        self._dynamic_toolsets = []
        self._toolset_by_agent: dict[CollabAgent, FunctionToolset] = defaultdict(FunctionToolset)
        self._handoffs = defaultdict(tuple)
        self._allow_back_handoff = allow_back_handoff
        self._already_handed_off = set[CollabAgent]()

        self._enter_lock = Lock()
        self._entered_count = 0
        self._exit_stack = None
        self._handoff_model = get_right_handoff_model(self._collab_settings)
        # Build internal structures
        self._build_topology()

        self._validate_agents_have_needed_attrs()
        self._validate_handoff_topology()

    def _get_explicit_handoff_model(self, handoff_agents: Sequence[CollabAgent]) -> type[HandOffBase[Any]]:
        from pydantic import Field

        if not handoff_agents:
            raise ValueError('No handoff agents specified, nothing to do with handoff model...')

        class ExplicitHandoffModel(self._handoff_model):
            next_agent: Literal[tuple(ho.name for ho in handoff_agents)] = Field(
                description='The next agent to hand off to'
            )

        return ExplicitHandoffModel

    def _normalize_agent(
        self,
        agent: AbstractAgent | tuple[AbstractAgent, str | None] | CollabAgent | str,
        require_description: bool = False,
    ) -> CollabAgent:
        """Convert various agent representations to CollabAgent.

        This is the single method for normalizing agent inputs throughout the Collab.

        Args:
            agent: Agent specified as:
                - CollabAgent: returned as-is
                - tuple[AbstractAgent, str | None]: wrapped with description
                - AbstractAgent: wrapped with no description
                - str: looked up in _name_to_agent dict
            require_description: If True, raise CollabError when description is missing.

        Returns:
            CollabAgent: The normalized agent wrapper.

        Raises:
            CollabError: If agent name not found, or description required but missing.
        """
        if isinstance(agent, CollabAgent):
            return agent

        if isinstance(agent, str):
            if not self._name_to_agent:
                raise CollabError(f"Cannot resolve agent name '{agent}' - Collab not yet initialized")
            if agent not in self._name_to_agent:
                raise CollabError(f"Agent '{agent}' not found in Collab")
            return self._name_to_agent[agent]

        if isinstance(agent, tuple):
            pydantic_agent, description = agent
            if require_description and description is None:
                raise CollabError(f"Description required for agent '{pydantic_agent.name}'")
            return CollabAgent(agent=pydantic_agent, description=description)

        # AbstractAgent
        if require_description:
            raise CollabError(f"Description required for agent '{agent.name}'")
        return CollabAgent(agent=agent, description=None)

    def _normalize_and_reuse_agent(
        self,
        agent: AbstractAgent | tuple[AbstractAgent, str | None] | CollabAgent | str,
    ) -> CollabAgent:
        """Normalize agent and reuse existing CollabAgent from _agents if possible.

        This method ensures object identity consistency by checking if the normalized
        agent's underlying AbstractAgent already exists in self._agents, and if so,
        returns that existing CollabAgent wrapper instead of creating a new one.

        Args:
            agent: Agent to normalize, can be string name, CollabAgent, tuple, or AbstractAgent.

        Returns:
            CollabAgent: Either an existing agent from _agents or a newly normalized one.
        """
        # Handle string lookups directly - already returns correct instance
        if isinstance(agent, str):
            return self._name_to_agent[agent]

        # Normalize the agent
        normalized = self._normalize_agent(agent)

        # Check if we already have this agent in _agents by matching underlying agent
        for existing_agent in self._agents:
            if existing_agent.agent is normalized.agent:
                return existing_agent

        # Not found in _agents, use the normalized version
        return normalized

    def __repr__(self) -> str:
        """Return a comprehensive string representation of the Collab."""
        return (
            f'{type(self).__name__}('
            f'agents={len(self._agents or [])}, '
            f'x={self.starting_agent.name if self.starting_agent else "unknown"!r}, '
            f'final_agent={self.final_agent.name!r}, '
            f'max_iterations={self.max_handoffs}'
            f')'
        )

    async def __aenter__(self) -> Self:
        """Enter the Collab context.

        Initializes resources needed for the Collb execution. Can be called
        multiple times (reference counted).

        Returns:
            Self: The Collab instance.
        """
        async with self._enter_lock:
            if self._entered_count == 0:
                self._exit_stack = AsyncExitStack()
                await self._exit_stack.__aenter__()
                # Resources initialized here if needed in the future
            self._entered_count += 1
        return self

    async def __aexit__(self, *args: Any) -> bool | None:
        """Exit the Collab context.

        Cleans up resources when the last reference exits.

        Args:
            *args: Exception information if an exception occurred.

        Returns:
            None to propagate exceptions, or True to suppress them.
        """
        async with self._enter_lock:
            self._entered_count -= 1
            if self._entered_count == 0 and self._exit_stack:
                await self._exit_stack.__aexit__(*args)
                self._exit_stack = None
        return None

    @property
    def agent_names(self) -> tuple[str, ...]:
        """Get the names of all agents in the Collab.

        Returns:
            tuple[str, ...]: Tuple of agent names.
        """
        return tuple(agent.name for agent in self._agents)

    @property
    def connections(self) -> dict[str, tuple[str, ...]]:
        """Get the agent-to-agent tool call connections.

        Returns:
            dict[str, tuple[str, ...]]: Mapping of agent names to callable agent names.
        """
        return {agent.name: tuple(conn.name for conn in conns) for agent, conns in self._agent_tools.items()}

    @property
    def handoffs(self) -> dict[str, tuple[str, ...]]:
        """Get the agent-to-agent handoff connections.

        Returns:
            dict[str, tuple[str, ...]]: Mapping of agent names to handoff target names.
        """
        return {agent.name: tuple(ho.name for ho in handoff_list) for agent, handoff_list in self._handoffs.items()}

    @property
    def has_handoffs(self) -> bool:
        return any(self.handoffs.values())

    def _get_name_from_agentdesc(self, agent_desc: t_agent_desc) -> str:
        if isinstance(agent_desc, (CollabAgent, AbstractAgent)):
            return agent_desc.name
        return agent_desc

    def _ensure_agent_present(self, agent: CollabAgent | None, msg: str = 'Agent must be present') -> CollabAgent:
        if agent is None:
            raise CollabError(msg)
        return agent

    def get_topology_ascii(self) -> str:
        """Return compact ASCII representation of the Collab topology.

        Format is optimized for LLM readability and token efficiency.

        Returns:
            str: ASCII diagram showing agents and shandoffs.

        Example output:
            Agents: Router, Researcher, Writer, Editor
            Start: Router | Final: Editor
            Handoffs (→): Router→Writer, Writer→Editor
        """
        if not self.has_handoffs:
            return ''
        # Agent list
        agent_names = ', '.join(a.name for a in self._agents)
        start = self._ensure_agent_present(self.starting_agent, 'No starting agent set')
        final = self._ensure_agent_present(self.final_agent, 'No final agent set')
        lines = [f'Agents: {agent_names}', f'Start: {start.name}, Final: {final.name}']
        # Collect handoffs
        handoff_pairs: list[str] = []
        for agent in self._agents:
            for target in self._handoffs.get(agent, ()):
                handoff_pairs.append(f'{agent.name}→{target.name}')

        if handoff_pairs:
            lines.append(f'Handoffs (→): {", ".join(handoff_pairs)}')
        else:
            lines.append('Handoffs: none')
        return '\n'.join(lines)

    def visualize_topology(
        self,
        figsize: tuple[float, float] = (12, 8),
        show: bool = True,
        save_path: str | None = None,
        title: str | None = None,
    ) -> 'matplotlib.figure.Figure':
        """Create a visual graph of the agent topology.

        Shows which agents can call which others as tools (dashed blue arrows)
        and handoffs (solid red arrows). Starting agent is highlighted in green,
        final agent in pink/red.

        By default, opens the plot in a window automatically (like matplotlib's .plot()).
        Set show=False to prevent automatic display.

        Args:
            figsize: Figure size as (width, height) in inches.
            show: Whether to display the figure immediately in a window. Defaults to True.
            save_path: Optional path to save the figure (e.g., 'topology.png').
            title: Optional title for the graph. Defaults to Collab name or 'Agent Topology'.

        Returns:
            matplotlib.figure.Figure: The generated figure object.

        Raises:
            ImportError: If networkx or matplotlib are not installed.

        Examples:
            # Automatically opens in a window (default behavior)
            collab.visualize_topology()

            # Save without showing
            collab.visualize_topology(save_path='my_collab.png', show=False)

            # Both save and show
            collab.visualize_topology(save_path='my_collab.png', show=True)
        """
        from ._viz import render_topology

        return render_topology(
            agents=self._agents,
            agent_tools=self._agent_tools,
            handoffs=self._handoffs,
            starting_agent=self.starting_agent,
            final_agent=self.final_agent,
            collab_name=self.name,
            figsize=figsize,
            show=show,
            save_path=save_path,
            title=title,
        )

    def _get_name_to_agent(self, name: str, exception: bool | str = True) -> CollabAgent | None:
        """Convert a name to an agent.

        Args:
            name: The name of the agent.
            exception: If True, raise an exception with generic message, if it's a string, raise the string.
                If false, return None
        """
        try:
            return self._name_to_agent[name]
        except KeyError:
            if exception is True:
                raise CollabError(f'Agent {name} not found in Collab.')
            elif exception:
                raise CollabError(exception)
            return None

    def _require_start_agent_is_final_agent(self, msg: str | None = None) -> None:
        """Validate that starting agent is the same as final agent.

        Args:
            msg: Optional custom error message.

        Raises:
            CollabError: If starting_agent is not the same as final_agent.
        """
        if self.starting_agent is not self.final_agent:
            raise CollabError(msg or 'This mode starting agent to be same as final agent')

    def _build_topology(self) -> None:
        """Build the agent topology from connections using edges.

        This method constructs the internal connection and handoff dictionaries
        based on explicit agent_calls
        and agent_handoffs.

        Raises:
            Exception: If starting agent has no path to final agent.
            Exception: If referenced agent names are not found.
        """
        # Register all agents from all connections
        if self.starting_agent is None:
            self.starting_agent = self._agents[0]
        elif self.starting_agent not in self._agents:
            self._name_to_agent[self.starting_agent.name] = self.starting_agent
            self._agents = (self.starting_agent, *self._agents)

        if self.final_agent is None:
            self.final_agent = self.starting_agent
        elif self.final_agent not in self._agents:
            self._name_to_agent[self.final_agent.name] = self.final_agent
            self._agents = (*self._agents, self.final_agent)

        for agent in self._agents:
            connections: list[CollabAgent] = []
            handoffs: list[CollabAgent] = []
            for called_agent_obj in agent.agent_calls:
                called_agent_collab = self._normalize_agent(called_agent_obj)
                if called_agent_collab is agent:
                    raise CollabError(f'Agent {called_agent_obj} cannot call itself!')
                connections.append(called_agent_collab)
            if self.final_agent is not self.starting_agent:
                for called_agent_obj in agent.agent_handoffs:
                    called_agent_collab = self._normalize_agent(called_agent_obj)
                    if called_agent_collab is agent:
                        raise CollabError(f'Agent {called_agent_obj} cannot call itself!')
                    handoffs.append(called_agent_collab)
            self._handoffs[agent] = tuple(handoffs)
            self._agent_tools[agent] = tuple(connections)

    def _validate_handoff_topology(self) -> None:
        """Validate that the handoff topology is sound.

        Ensures:
        1. Starting agent can reach final agent
        2. Every reachable agent can eventually reach final agent (no dead ends)
        3. No closed cycles exist without a path to final agent

        Raises:
            CollabError: If topology is invalid with detailed explanation.
        """
        if self.starting_agent is self.final_agent:
            return  # No handoffs needed, topology is trivially valid

        def reachable_from(
            start: CollabAgent, adjacency: dict[CollabAgent, Collection[CollabAgent]]
        ) -> set[CollabAgent]:
            """BFS to find all agents reachable from start via adjacency."""
            visited = {start}
            frontier = [start]
            while frontier:
                for neighbor in adjacency.get(frontier.pop(), ()):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        frontier.append(neighbor)
            return visited

        # Build reverse adjacency for backward traversal
        # Use defaultdict with explicit generic typing removed to appease pyright's
        # strict checks about runtime type expressions.
        reverse_handoffs = defaultdict(set[CollabAgent])
        for agent, targets in self._handoffs.items():
            for target in targets:
                reverse_handoffs[target].add(agent)

        # Forward: who can we reach from start?
        start_agent = self._ensure_agent_present(self.starting_agent, 'No starting agent set')
        final_agent = self._ensure_agent_present(self.final_agent, 'No final agent set')
        reachable = reachable_from(start_agent, self._handoffs)
        # Backward: who can reach final?
        can_reach_final = reachable_from(final_agent, cast(dict, reverse_handoffs))

        # Check 1: Can start reach final?
        if final_agent not in reachable:
            raise CollabError(
                f"Starting agent '{start_agent.name}' has no handoff path to "
                f"final agent '{final_agent.name}'. "
                f'Reachable agents: {sorted(a.name for a in reachable)}'
            )

        # Check 2: Every reachable agent must be able to reach final (catches dead ends + inescapable cycles)
        dead_ends = sorted(a.name for a in reachable - can_reach_final)
        if dead_ends:
            raise CollabError(
                f'Dead-end agents detected: {dead_ends}. '
                f'These agents can receive handoffs but have no path to final agent '
                f"'{final_agent.name}'."
            )

    def _is_final_agent(self, agent: CollabAgent) -> bool:
        """Check if an agent is a final agent (can output to user).

        Args:
            agent: The agent to check.

        Returns:
            bool: True if agent is the designated final agent.
        """
        return agent == self.final_agent

    def _get_allowed_handoff_targets(self, agent: CollabAgent) -> Sequence[CollabAgent]:
        if not self._handoffs.get(agent):
            return ()
        if not self._allow_back_handoff:
            return tuple(set(self._handoffs[agent]) - set(self._already_handed_off))
        return tuple(self._handoffs[agent])

    async def _run_agent(
        self,
        c_agent: CollabAgent,
        query: str,
        context_from_handoff: str | None,
        message_history: list[ModelRequest | ModelResponse] | None,
        state: CollabState,
        usage: RunUsage,
        is_tool_run: bool = False,
        max_agent_calls_depths_allowed: int | None = None,
        deps: dict[t_agent_name, AgentDepsT] | None = None,
    ) -> AgentRunSummary:
        """Run a single agent and get its output ( or HandoffOutput).

        Args:
            c_agent: The agent to run.
            query: The current query/prompt for this agent.
            context_from_handoff: Context passed from previous agent handoff.
            message_history: Previous message history for this agent.
            state: Current Collab execution state.
            usage: Accumulated usage statistics.
            is_tool_run: Whether the agent called as a tool call by another agent.
            max_agent_calls_depths_allowed: Depth of agents calling others as tools. resets after every handoff.
            deps: Dependencies Instance to use per Agent. Only used if agent has deps_type.

        Returns:
            AgentRunResult: Result containing output and message history.

        Raises:
            CollabError: If call_agent tries to route to unreachable agent.
        """
        if message_history and context_from_handoff:
            raise CollabError("Can't have both message_history and context_from_handoff")
        if max_agent_calls_depths_allowed is None:
            max_agent_calls_depths_allowed = self.max_agent_call_depth
        if deps is None:
            deps = {}
        # Determine c_agent capabilities
        is_final = self._is_final_agent(c_agent)

        # We just show it as though no tool agents exist if we get to depth 0
        tool_agents = self._agent_tools.get(c_agent, ()) if max_agent_calls_depths_allowed > 0 else ()

        # When it's a tool run we don't allow handoffs, it causes too much confusion
        if not is_tool_run:
            handoff_agents = self._get_allowed_handoff_targets(c_agent)
        else:
            handoff_agents = ()
        can_handoff = bool(handoff_agents)

        # Get the appropriate output type for this c_agent. Generally using pydantic here we make sure only relevant
        # Things can be called
        output_type = self.get_output_type_for_agent(c_agent, handoff_agents)

        # Tools don't need to know topology.
        if self._collab_settings.include_topology_in_prompt and not is_tool_run:
            topology = self.get_topology_ascii()
        else:
            topology = None
        # Build prompt using the configurable prompt builder
        prompt_ctx = PromptBuilderContext(
            agent=c_agent,
            final_agent=is_final,
            can_handoff=can_handoff,
            handoff_agents=handoff_agents,
            tool_agents=tool_agents,
            called_as_tool=is_tool_run,
            ascii_topology=topology,
        )
        prompt_builder = self._collab_settings.prompt_builder or default_build_agent_prompt
        info_about_agents = prompt_builder(prompt_ctx)
        user_prompt = f'{info_about_agents}\n\n{context_from_handoff or ""}'.strip() + f'\n\n# Query: {query}'
        toolsets = (*self._toolsets, *self._dynamic_toolsets, self._toolset_by_agent[c_agent])
        if tool_agents:
            tool_call = self._get_agent_call_tool(c_agent, state, max_agent_calls_depths_allowed, deps)
            toolsets = (*toolsets, FunctionToolset((tool_call,)))

        self._already_handed_off.add(c_agent)

        # Run the c_agent with the constrained output type, check time it takes to run
        start_time = datetime.datetime.now()

        # We do this to make sure MCPs are initialized.
        # TODO: prevent doing a lot of entering into an agent called more than once.
        async with c_agent.agent as agent:
            run_result: AgentRunResult[output_type] = await agent.run(
                user_prompt=user_prompt,
                deps=deps.get(c_agent.name) if c_agent.requires_deps else None,
                output_type=output_type,
                toolsets=toolsets,
                usage_limits=self.usage_limits,
                usage=usage,
                model=self.model,
                model_settings=self.model_settings,
                message_history=message_history,
                instructions=self.instructions,
            )
        run_time = time.time() - start_time.toordinal()
        output: output_type = run_result.output
        # Get output data
        result = AgentRunSummary(
            agent_name=c_agent.name,
            messages=run_result.all_messages(),
            start_time=start_time,
            run_time=run_time,
            output=output,
            usage=run_result.usage(),
        )
        state.messages.append(result)
        return result

    async def run(self, query: str, deps: AgentDepsT | dict[t_agent_desc, AgentDepsT] | None = None) -> CollabRunResult:
        """Run the Collab with the given query.

        Executes the Collab starting from the starting_agent and following
        handoffs until a final agent returns FinalOutput or max_iterations
        is reached.

        Args:
            query: The user's query to process.
            deps: Either an AgentDepsT Instance, which will be used for all agents that require _deps, or a dictionary
            mapping  agent names to AgentDepsT instances. Defaults to None - no Deps will be used.

        Returns:
            CollabRunResult
        Raises:
            CollabError: If an agent tries to hand off to invalid agent.
        """
        if self._logfire is not None:
            span = self._logfire.span(f'{self.name or "Agent"} Collab Run').__enter__()
        else:
            span = None
        try:
            deps_parsed: dict[t_agent_name, AgentDepsT] = {}
            if isinstance(deps, dict):
                dep_obj: Any
                for desc, dep_obj in deps.items():
                    deps_parsed[self._get_name_from_agentdesc(desc)] = dep_obj
            elif deps is not None:
                deps_parsed = {a.name: deps for a in self._agents}

            state = CollabState(query=query)
            usage = RunUsage()
            # Start with the starting agent
            current_agent_name = self.starting_agent.name
            current_agent = self.starting_agent
            current_query: Any = query

            # Handoff options (default for first agent)
            output: AgentRunSummary | None = None
            handoff_count = -1  # Track actual iterations
            # Defensive cap to avoid runaway loops causing excessive model requests
            handoff_data_str = ''
            context_builder_func = self._collab_settings.context_builder or get_context
            self._logger.info(f'Staring collab: calling agent {current_agent_name}.')
            while handoff_count < self.max_handoffs:
                handoff_count += 1
                # Track execution path
                state.execution_path.append(current_agent_name)

                # Run current agent
                output: AgentRunSummary = await self._run_agent(
                    current_agent, current_query, handoff_data_str, [], state, usage=usage, deps=deps_parsed
                )
                usage, output_data = output.usage, output.output
                state.record_execution(current_agent_name, current_query, output_data)

                if not isinstance(output_data, HandOffBase):
                    state.final_output = output_data
                    break

                # Here we assume output_data is HandOff
                # Verify the handoff target is a known agent
                new_agent = self._get_name_to_agent(
                    output_data.next_agent, f'Invalid agent name: {output_data.next_agent}'
                )

                # Enforce declared handoff permissions: an agent may only hand off to
                # targets listed in its `agent_handoffs`. This prevents agents
                # from dynamically handing off to arbitrary agents and helps avoid
                # runaway loops caused by unexpected routing.
                # Generallt it's unnecessary as pydantic already enforces it
                if new_agent not in self._get_allowed_handoff_targets(current_agent):
                    raise CollabError(f"Can't hand off agent {new_agent} to invalid handoff target.")
                if self._logger is not None:
                    self._logger.info(f'{current_agent_name} is handing off to {output_data.next_agent}.')

                hod = HandoffData(
                    previous_handoff_str=handoff_data_str if output_data.include_previous_handoff else None,
                    callee_agent_name=new_agent.name,
                    caller_agent_name=current_agent_name,
                    message_history=output.messages if output_data.include_conversation else None,
                    include_thinking=output_data.include_thinking,
                    include_tool_calls_with_callee=output_data.include_tool_calls_with_callee,
                )
                handoff_data_str = context_builder_func(hod)
                current_agent = new_agent
                current_agent_name = current_agent.name
                current_query = output_data.query

            return CollabRunResult(
                output=output.output,
                usage=usage,
                iterations=handoff_count + 1,
                final_agent=current_agent_name,
                max_iterations_reached=(handoff_count >= self.max_handoffs and isinstance(output.output, HandOffBase)),
                _state=state,
            )
        except Exception as e:
            self._logger.error(f'Collab failed with exception: {e}')
        finally:
            if span is not None:
                span.__exit__(None, None, None)

    def get_output_type_for_agent(
        self,
        agent: CollabAgent,
        handoff_agents: Sequence[CollabAgent] | None,
    ) -> type[HandOffBase[OutputDataT]] | OutputDataT:
        """Get the appropriate output type based on agent capabilities.

        Args:
            agent: The agent to determine output type for.
            handoff_agents: List of agents it can handoff to, None or empty list if not handoffs.

        Returns:
            type[HandOffBase] | type[FinalOutput]: The appropriate Pydantic model
                type for the agent's output. HandoffOutput if agent can handoff,
                FinalOutput otherwise. The inner type is determined by
                _collab_settings.output_restrictions.
        """
        agent_ot = agent.agent.output_type if self._output_type is None else self._output_type
        if self._collab_settings.output_restrictions == 'only_str':
            inner_type = str
        elif self._collab_settings.output_restrictions == 'only_original':
            inner_type = agent_ot
        else:
            inner_type = str | agent_ot
        if handoff_agents:
            our_handoff_model = self._get_explicit_handoff_model(handoff_agents)
            return our_handoff_model[inner_type]
        return inner_type

    def _get_agent_call_tool(
        self,
        caller: CollabAgent,
        state: CollabState,
        max_agent_calls_depths_allowed: int,
        deps: dict[t_agent_name, AgentDepsT],
    ) -> Tool:
        """This function is used to get the Tool that's used to use agent calls between agents.

        Args:
            caller: The agent that can call this tool
            state: The current state of execution. Used just to send to _run_agent when called
            usage: usage statistics of the tool for tracking, just forwarded to _run_agent
            max_agent_calls_depths_allowed: How many depths of agent calls are still allowed. Decremented when tool is
                called
            deps: a dictionary of dependencies, just forwarded to _run_agent
        Returns:
            Tool that's used to use agent calls between agents
        """
        my_mem = {}
        callable_agents = tuple(agent.name for agent in self._agent_tools[caller])

        async def call_agent(  # noqa: D417
            ctx: RunContext,
            agent_name: Literal[callable_agents],
            input: Any,
            keep_memory_from_before: bool = True,
        ) -> Any:
            """Call agent and return response.

            Args:
                agent_name: The name of the agent to call, only out of the list you have
                input: The input for the agent
                keep_memory_from_before: Boolean, whether to let the agent have data from previous calls to it.
                Defaults to True
            Returns:
                The output of the agent
            """
            next_agent = self._get_name_to_agent(
                agent_name, f"{caller.name} tried to route to {agent_name} but it's not reachable. "
            )

            # This is not really necessary as pydantic-ai already verifies that using the LIteral. Keeping because we
            # might change chat.
            if next_agent not in self._agent_tools[caller]:
                raise CollabError(f"Can't call agent {agent_name} from {caller.name} - not in allowed connections")

            # We don't count this tool in usage. TODO: create a separate usage statsic for it
            ctx.usage.tool_calls -= 1
            inner_message_history = []
            if keep_memory_from_before and next_agent in my_mem:
                inner_message_history = my_mem[next_agent]
            if self._logger is not None:
                self._logger.info(f'Calling agent {agent_name} from {caller.name}')
            try:
                resp = await self._run_agent(
                    next_agent,
                    input,
                    None,
                    inner_message_history,
                    state,
                    ctx.usage,
                    is_tool_run=True,
                    max_agent_calls_depths_allowed=max_agent_calls_depths_allowed - 1,
                    deps=deps,
                )
            # We can't really catch an exception in after agent handoff and then hand it off to some other agent,
            # But it's possibles to do that here, that's why the try except is not inside run_agent but rather here
            # Errors here should generally be errors from tools the other Agent ran.
            except Exception as e:
                self._logger.error(f'Error calling agent {agent_name} from {caller.name}: {e}')
                raise ModelRetry(
                    f'Calling agent {agent_name} failed unexpectedly with error {e}. '
                    'decide according to the error how to proceed'
                )

            # TODO: deal with parallel tool calls to the same agent. Maybe forbid it if memory is set?
            if keep_memory_from_before or my_mem.get(next_agent, None) is None:
                my_mem[next_agent] = resp.messages
            return resp.output

        # The use of sequential here is problematic in the sense that it does not allow this call to be in parallel
        # to any other call. TODO: fix that with some lock or more internal pydantic_ai mech?
        call_agent_tool = Tool(call_agent, takes_ctx=True, sequential=not self._allow_parallel_agent_calls)
        return call_agent_tool

    def run_sync(self, query: str, deps: dict[t_agent_desc, AgentDepsT] | AgentDepsT | None = None) -> CollabRunResult:
        """Run the Collab with the given query.

        Executes the Collab starting from the starting_agent and following
        handoffs until a final agent returns FinalOutput or max_iterations
        is reached.

        Args:
            query: The user's query to process.
            deps: Either an AgentDepsT Instance, which will be used for all agents that require _deps, or a dictionary
            mapping  agent names to AgentDepsT instances. Defaults to None - no Deps will be used.

        Returns:
            CollabRunResult
        Raises:
            CollabError: If an agent tries to hand off to invalid agent.
        """
        import asyncio

        return asyncio.run(self.run(query, deps=deps))

    def _get_ts_by_agents(
        self, agents: Collection[str | CollabAgent | AbstractAgent] | str | CollabAgent | AbstractAgent | None
    ) -> Collection[FunctionToolset]:
        """Return FunctionToolset(s) for given agent descriptor(s).

        Accepts:
        - None -> returns the default user toolset tuple
        - str or CollabAgent or AbstractAgent -> returns a single-item tuple
        - Collection[...] -> returns a tuple of toolsets in the same order
        """
        if agents is None:
            return (self._user_toolset,)
        # single name
        if isinstance(agents, str):
            return (self._toolset_by_agent[self._normalize_agent(agents)],)
        # single agent object
        if isinstance(agents, (CollabAgent, AbstractAgent)):
            return (self._toolset_by_agent[self._normalize_agent(agents)],)
        # iterable of agent descriptors
        return tuple(self._toolset_by_agent[self._normalize_agent(agent)] for agent in agents)

    @property
    def is_instrumented(self) -> bool:
        return any(getattr(agent, '_instrument_default', False) for agent in self._agents)

    @overload
    def tool(self, func: ToolFuncContext[AgentDepsT, ToolParams], /) -> ToolFuncContext[AgentDepsT, ToolParams]: ...

    @overload
    def tool(
        self,
        /,
        *,
        agents: str | Collection[str | CollabAgent] | None = None,
        name: str | None = None,
        description: str | None = None,
        retries: int | None = None,
        prepare: ToolPrepareFunc[AgentDepsT] | None = None,
        docstring_format: DocstringFormat = 'auto',
        require_parameter_descriptions: bool = False,
        schema_generator: type[GenerateJsonSchema] = GenerateToolJsonSchema,
        strict: bool | None = None,
        sequential: bool = False,
        requires_approval: bool = False,
        metadata: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> Callable[[ToolFuncContext[AgentDepsT, ToolParams]], ToolFuncContext[AgentDepsT, ToolParams]]: ...

    def tool(
        self,
        func: ToolFuncContext[AgentDepsT, ToolParams] | None = None,
        /,
        *,
        agents: str | Collection[str | CollabAgent] | None = None,
        name: str | None = None,
        description: str | None = None,
        retries: int | None = None,
        prepare: ToolPrepareFunc[AgentDepsT] | None = None,
        docstring_format: DocstringFormat = 'auto',
        require_parameter_descriptions: bool = False,
        schema_generator: type[GenerateJsonSchema] = GenerateToolJsonSchema,
        strict: bool | None = None,
        sequential: bool = False,
        requires_approval: bool = False,
        metadata: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> Any:
        """Decorator to register a tool function which takes [`RunContext`][pydantic_ai.tools.RunContext] as its first argument.

        Can decorate a sync or async functions.

        The docstring is inspected to extract both the tool description and description of each parameter,
        [learn more](../tools.md#function-tools-and-schema).

        We can't add overloads for every possible signature of tool, since the return type is a recursive union
        so the signature of functions decorated with `@agent.tool` is obscured.

        Args:
            func: The tool function to register.
            agents: One or more names of agents to register the tool for. By default all agents are registered.,
            name: The name of the tool, defaults to the function name.
            description: The description of the tool, defaults to the function docstring.
            retries: The number of retries to allow for this tool, defaults to the agent's default retries,
                which defaults to 1.
            prepare: custom method to prepare the tool definition for each step, return `None` to omit this
                tool from a given step. This is useful if you want to customise a tool at call time,
                or omit it completely from a step. See [`ToolPrepareFunc`][pydantic_ai.tools.ToolPrepareFunc].
            docstring_format: The format of the docstring, see [`DocstringFormat`][pydantic_ai.tools.DocstringFormat].
                Defaults to `'auto'`, such that the format is inferred from the structure of the docstring.
            require_parameter_descriptions: If True, raise an error if a parameter description is missing. Defaults to False.
            schema_generator: The JSON schema generator class to use for this tool. Defaults to `GenerateToolJsonSchema`.
            strict: Whether to enforce JSON schema compliance (only affects OpenAI).
                See [`ToolDefinition`][pydantic_ai.tools.ToolDefinition] for more info.
            sequential: Whether the function requires a sequential/serial execution environment. Defaults to False.
            requires_approval: Whether this tool requires human-in-the-loop approval. Defaults to False.
                See the [tools documentation](../deferred-tools.md#human-in-the-loop-tool-approval) for more info.
            metadata: Optional metadata for the tool. This is not sent to the model but can be used for filtering and tool behavior customization.
            timeout: Timeout in seconds for tool execution. If the tool takes longer, a retry prompt is returned to the model.
                Overrides the agent-level `tool_timeout` if set. Defaults to None (no timeout).
        """

        def tool_decorator(
            func_: ToolFuncContext[AgentDepsT, ToolParams],
        ) -> ToolFuncContext[AgentDepsT, ToolParams]:
            # noinspection PyTypeChecker
            toolsets = self._get_ts_by_agents(agents)
            for ts in toolsets:
                ts.add_function(
                    func_,
                    takes_ctx=True,
                    name=name,
                    description=description,
                    retries=retries,
                    prepare=prepare,
                    docstring_format=docstring_format,
                    require_parameter_descriptions=require_parameter_descriptions,
                    schema_generator=schema_generator,
                    strict=strict,
                    sequential=sequential,
                    requires_approval=requires_approval,
                    metadata=metadata,
                    timeout=timeout,
                )
            return func_

        return tool_decorator if func is None else tool_decorator(func)

    @overload
    def tool_plain(self, func: ToolFuncPlain[ToolParams], /) -> ToolFuncPlain[ToolParams]: ...

    @overload
    def tool_plain(
        self,
        /,
        *,
        agents: str | Collection[str | CollabAgent] | None = None,
        name: str | None = None,
        description: str | None = None,
        retries: int | None = None,
        prepare: ToolPrepareFunc[AgentDepsT] | None = None,
        docstring_format: DocstringFormat = 'auto',
        require_parameter_descriptions: bool = False,
        schema_generator: type[GenerateJsonSchema] = GenerateToolJsonSchema,
        strict: bool | None = None,
        sequential: bool = False,
        requires_approval: bool = False,
        metadata: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> Callable[[ToolFuncPlain[ToolParams]], ToolFuncPlain[ToolParams]]: ...

    def tool_plain(
        self,
        func: ToolFuncPlain[ToolParams] | None = None,
        /,
        *,
        agents: str | Collection[str | CollabAgent] | None = None,
        name: str | None = None,
        description: str | None = None,
        retries: int | None = None,
        prepare: ToolPrepareFunc[AgentDepsT] | None = None,
        docstring_format: DocstringFormat = 'auto',
        require_parameter_descriptions: bool = False,
        schema_generator: type[GenerateJsonSchema] = GenerateToolJsonSchema,
        strict: bool | None = None,
        sequential: bool = False,
        requires_approval: bool = False,
        metadata: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> Any:
        """Decorator to register a tool function which DOES NOT take `RunContext` as an argument.

        Can decorate a sync or async functions.

        The docstring is inspected to extract both the tool description and description of each parameter,
        [learn more](../tools.md#function-tools-and-schema).

        We can't add overloads for every possible signature of tool, since the return type is a recursive union
        so the signature of functions decorated with `@agent.tool` is obscured.

        Example:
        ```python
        from pydantic_ai import Agent, RunContext

        agent = Agent('test')

        @agent.tool
        def foobar(ctx: RunContext[int]) -> int:
            return 123

        @agent.tool(retries=2)
        async def spam(ctx: RunContext[str]) -> float:
            return 3.14

        result = agent.run_sync('foobar', _deps=1)
        print(result.output)
        #> {"foobar":123,"spam":3.14}
        ```

        Args:
            func: The tool function to register.
            agents: One or more names of agents to register the tool for. By default all agents are registered.,
            name: The name of the tool, defaults to the function name.
            description: The description of the tool, defaults to the function docstring.
            retries: The number of retries to allow for this tool, defaults to the agent's default retries,
                which defaults to 1.
            prepare: custom method to prepare the tool definition for each step, return `None` to omit this
                tool from a given step. This is useful if you want to customise a tool at call time,
                or omit it completely from a step. See [`ToolPrepareFunc`][pydantic_ai.tools.ToolPrepareFunc].
            docstring_format: The format of the docstring, see [`DocstringFormat`][pydantic_ai.tools.DocstringFormat].
                Defaults to `'auto'`, such that the format is inferred from the structure of the docstring.
            require_parameter_descriptions: If True, raise an error if a parameter description is missing. Defaults to False.
            schema_generator: The JSON schema generator class to use for this tool. Defaults to `GenerateToolJsonSchema`.
            strict: Whether to enforce JSON schema compliance (only affects OpenAI).
                See [`ToolDefinition`][pydantic_ai.tools.ToolDefinition] for more info.
            sequential: Whether the function requires a sequential/serial execution environment. Defaults to False.
            requires_approval: Whether this tool requires human-in-the-loop approval. Defaults to False.
                See the [tools documentation](../deferred-tools.md#human-in-the-loop-tool-approval) for more info.
            metadata: Optional metadata for the tool. This is not sent to the model but can be used for filtering and tool behavior customization.
            timeout: Timeout in seconds for tool execution. If the tool takes longer, a retry prompt is returned to the model.
                Overrides the agent-level `tool_timeout` if set. Defaults to None (no timeout).
        """

        def tool_decorator(func_: ToolFuncPlain[ToolParams]) -> ToolFuncPlain[ToolParams]:
            # noinspection PyTypeChecker
            toolsets = self._get_ts_by_agents(agents)
            for ts in toolsets:
                ts.add_function(
                    func_,
                    takes_ctx=False,
                    name=name,
                    description=description,
                    retries=retries,
                    prepare=prepare,
                    docstring_format=docstring_format,
                    require_parameter_descriptions=require_parameter_descriptions,
                    schema_generator=schema_generator,
                    strict=strict,
                    sequential=sequential,
                    requires_approval=requires_approval,
                    metadata=metadata,
                    timeout=timeout,
                )
            return func_

        return tool_decorator if func is None else tool_decorator(func)

    @overload
    def toolset(self, func: ToolsetFunc[AgentDepsT], /) -> ToolsetFunc[AgentDepsT]: ...

    @overload
    def toolset(
        self,
        /,
        *,
        per_run_step: bool = True,
    ) -> Callable[[ToolsetFunc[AgentDepsT]], ToolsetFunc[AgentDepsT]]: ...

    def toolset(
        self,
        func: ToolsetFunc[AgentDepsT] | None = None,
        /,
        *,
        per_run_step: bool = True,
    ) -> Any:
        """Decorator to register a toolset function which takes [`RunContext`][pydantic_ai.tools.RunContext] as its only argument.

        Can decorate a sync or async functions.

        The decorator can be used bare (`agent.toolset`).

        Example:
        ```python
        from pydantic_ai import AbstractToolset, Agent, FunctionToolset, RunContext

        agent = Agent('test', deps_type=str)

        @agent.toolset
        async def simple_toolset(ctx: RunContext[str]) -> AbstractToolset[str]:
            return FunctionToolset()
        ```

        Args:
            func: The toolset function to register.
            per_run_step: Whether to re-evaluate the toolset for each run step. Defaults to True.
        """
        from pydantic_ai.toolsets._dynamic import DynamicToolset

        def toolset_decorator(func_: ToolsetFunc[AgentDepsT]) -> ToolsetFunc[AgentDepsT]:
            self._dynamic_toolsets.append(DynamicToolset(func_, per_run_step=per_run_step))
            return func_

        return toolset_decorator if func is None else toolset_decorator(func)

    def _init_logfire(self):
        """Initialize logfire.

        Initialize only if:
        1. _instrument_logfire is True
        2. logfire.configure() has been already called.
        """
        self._logfire = None
        if self._instrument_logfire:
            try:
                import logfire
            except ImportError:
                logfire = None
            else:
                try:
                    configured = logfire.DEFAULT_LOGFIRE_INSTANCE.config._initialized
                except AttributeError:
                    configured = False
                logfire = logfire if configured else None
            self._logfire = logfire
        if self._logfire is None:
            import logging

            self._logger = logging.getLogger('pydantic_collab')
        else:
            self._logger = self._logfire

    def _validate_agents_have_needed_attrs(self):
        if len(self._name_to_agent) != len(self._agents):
            raise ValueError('You cannot use same Agent name for different agents')

        # Verify all agents have description except starting agent that doesn't need if it's not a tool

        for agent in self._agents:
            if not agent.description:
                if agent is self.starting_agent:
                    if agent in (i for conns in self._agent_tools.values() for i in conns):
                        raise ValueError(
                            f'Starting Agent {agent.name} needs to have description as it '
                            'can be called by another agent'
                        )
                    elif agent in (i for conns in self._handoffs.values() for i in conns):
                        raise ValueError(
                            f'Starting Agent {agent.name} needs to have description as it '
                            'can be hand-offed to by another agent'
                        )
                else:
                    raise ValueError(f"Agent {agent.name} must have a description, as it's not the starting agent")
