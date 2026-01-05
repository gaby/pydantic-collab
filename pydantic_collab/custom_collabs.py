from collections.abc import Sequence
from dataclasses import field

from pydantic_ai import ModelSettings, Tool, UsageLimits
from pydantic_ai.agent.abstract import AbstractAgent, Instructions
from pydantic_ai.models import KnownModelName, Model
from pydantic_ai.tools import ToolFuncEither

from ._types import AgentDepsT, CollabAgent, CollabError, CollabSettings, OutputDataT
from .collab import Collab


class StarCollab(Collab[AgentDepsT, OutputDataT]):
    """Collaboration of agents where there are no handoffs and only tool calls.

    router_agent (or the first agent in ´agents´ if not specified) is the starting and final agent and can call other
    agents as tools
    """

    def __init__(
        self,
        agents: Sequence[CollabAgent | tuple[AbstractAgent, str | None]],
        router_agent: CollabAgent | AbstractAgent | tuple[AbstractAgent, str | None] | None = None,
        name: str | None = None,
        output_type: OutputDataT | None = None,
        tools: Sequence[Tool | ToolFuncEither[AgentDepsT, ...]] | None = None,
        model: Model | KnownModelName | str | None = None,
        model_settings: ModelSettings | None = None,
        collab_settings: CollabSettings | None = None,
        usage_limits: UsageLimits | None = None,
        instructions: Instructions[AgentDepsT] = None,
        max_agent_call_depth: int = 3,
        instrument_logfire: bool = True,
    ) -> None:
        super().__init__(
            agents,
            router_agent,
            final_agent=None,
            name=name,
            output_type=output_type,
            tools=tools,
            model=model,
            model_settings=model_settings,
            collab_settings=collab_settings,
            usage_limits=usage_limits,
            instructions=instructions,
            max_agent_call_depth=max_agent_call_depth,
            max_handoffs=0,
            instrument_logfire=instrument_logfire,
        )

    def _build_topology(self):
        self._handoffs = {}
        if self.starting_agent is None:
            # default to first agent
            if not self._agents:
                raise CollabError('No agents available to set as starting_agent')
            self.starting_agent = self._agents[0]
        self._agent_tools[self.starting_agent] = tuple(i for i in self._agents if i is not self.starting_agent)
        if self.final_agent not in (self.starting_agent, None):
            raise CollabError(f'Final Agent must be either None or starting_agent in {self.__class__.__name__}')
        self.final_agent = self.starting_agent


class MeshCollab(Collab[AgentDepsT, OutputDataT]):
    """Mesh of Collaboration agents - each agent can call other agents as tools. No handoff happens."""

    def _build_topology(self):
        self._handoffs = {}
        if self.starting_agent is None:
            if not self._agents:
                raise CollabError('No agents available to set as starting_agent')
            self.starting_agent = self._agents[0]
        self._agent_tools[self.starting_agent] = tuple(i for i in self._agents if i is not self.starting_agent)
        for agent in self._agents:
            if agent is not self.starting_agent:
                # Mesh: all agents can call all other agents (including starting agent)
                self._agent_tools[agent] = tuple(i for i in self._agents if i is not agent)
        if self.final_agent not in (self.starting_agent, None):
            raise CollabError(f'Final Agent must be either None or starting_agent in {self.__class__.__name__}')
        self.final_agent = self.starting_agent


class PiplineCollab(Collab[AgentDepsT, OutputDataT]):
    """Pipeline Collab - agents[0] hands off to agent[1] hands off to agents[2].

    Agents can and should hand off to the next agent, according to the order of them supplied to ´agents´.
    """

    def _build_topology(self):
        if self.starting_agent is None:
            self.starting_agent = self._agents[0]
        if self.final_agent is None:
            self.final_agent = self._agents[-1]
        elif self.final_agent not in self._agents:
            self._agents = (*self._agents, self.final_agent)
        elif self.final_agent != self._agents[-1]:
            raise CollabError('Final agent must be last agent when using PiplineCollab')
        if self._agents.index(self.starting_agent) != 0:
            raise CollabError('When using PiplineCollab, starting agent must be first or not present in agents')
        for ag_num in range(len(self._agents) - 1):
            self._handoffs[self._agents[ag_num]] = (self._agents[ag_num + 1],)


class HierarchyCollab(Collab[AgentDepsT, OutputDataT]):
    """Hierarchy Collab."""

    _planner: CollabAgent = field(repr=False)
    _orchestrator_agents: Sequence[CollabAgent] = field(repr=False)

    def __init__(
        self,
        planner: AbstractAgent | CollabAgent | tuple[AbstractAgent, str | None],
        orchestrator_agents: dict[
            CollabAgent | tuple[AbstractAgent, str], list[CollabAgent | tuple[AbstractAgent, str] | Tool]
        ],
        router_agent: CollabAgent | AbstractAgent | tuple[AbstractAgent, str | None] | None = None,
        name: str | None = None,
        output_type: OutputDataT | None = None,
        tools: Sequence[Tool | ToolFuncEither[AgentDepsT, ...]] | None = None,
        model: Model | KnownModelName | str | None = None,
        model_settings: ModelSettings | None = None,
        collab_settings: CollabSettings | None = None,
        usage_limits: UsageLimits | None = None,
        instructions: Instructions[AgentDepsT] = None,
        max_agent_call_depth: int = 3,
        instrument_logfire: bool = True,
    ) -> None:
        self._planner = self._normalize_agent(planner)
        # Normalize orchestrator mapping: keys -> CollabAgent, values -> list[CollabAgent|Tool]
        normalized_map: dict[CollabAgent, list[CollabAgent | Tool]] = {}
        for key, vals in orchestrator_agents.items():
            norm_key = self._normalize_agent(key)
            norm_vals: list[CollabAgent | Tool] = []
            for v in vals:
                # Tools should be preserved as-is
                if isinstance(v, Tool):
                    norm_vals.append(v)
                else:
                    norm_vals.append(self._normalize_agent(v))
            normalized_map[norm_key] = norm_vals
        # Store both a sequence of orchestrator agents and the mapping for use in topology building
        self._orchestrator_agents = list(normalized_map.keys())
        self._orchestrator_map = normalized_map
        super().__init__(
            [],
            router_agent,
            final_agent=None,
            name=name,
            output_type=output_type,
            tools=tools,
            model=model,
            model_settings=model_settings,
            collab_settings=collab_settings,
            usage_limits=usage_limits,
            instructions=instructions,
            max_agent_call_depth=max_agent_call_depth,
            max_handoffs=0,
            instrument_logfire=instrument_logfire,
        )

    def _build_topology(self):
        pass
