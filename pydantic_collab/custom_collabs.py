from collections.abc import Sequence

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
        allow_parallel_agent_calls: bool = True,
        instrument_logfire: bool = True,
    ) -> None:
        """Initiate a Collab.

        Args:
            agents: A list of agents with calls and handoffs specified. Agents that don't have calls can be specified
                as (pydantic_ai.Agent, <description>), Agent that doesn't need description can be specified pydantic_ai.Agent
            router_agent: The starting and final agent that can call other agents as tools
            name: Optional name for the Collab, used for logging
            output_type: Expected output type; overrides the final_agent's output type if specified,
            tools: tools to add to all agents, in addition to any tools already specified for the agents.
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
            instrument_logfire: Whether to instruments Logfire. If true, logfire will be used if it can be imported and
                has *already* been configured.
        """
        super().__init__(agents,
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
                         allow_parallel_agent_calls=allow_parallel_agent_calls,
                         instrument_logfire=instrument_logfire)

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
        if self.final_agent  not in (self.starting_agent, None):
            raise CollabError(f'Final Agent must be either None or starting_agent in {self.__class__.__name__}')
        self.final_agent = self.starting_agent


class ForwardHandoffCollab(Collab[AgentDepsT, OutputDataT]):
    """Forward handoff Collab.

    Agents can and should hand off to the next agent, according to the order of them supplied to ´agents´.
    No Agent Toolcalls available.
    """

    def _build_topology(self):
        if self.starting_agent is None:
            self.starting_agent = self._agents[0]
        if self.final_agent is None:
            self.final_agent = self._agents[-1]
        elif self.final_agent not in self._agents:
            self._agents = (*self._agents, self.final_agent)
        elif self.final_agent != self._agents[-1]:
            raise CollabError('Final agent must be last agent when using ForwardHandoffCollab')
        if self._agents.index(self.starting_agent) != 0:
            raise CollabError(
                'When using ForwardHandoffCollab, starting agent must be first or not present in agents'
            )
        for ag_num in range(len(self._agents) - 1):
            self._handoffs[self._agents[ag_num]] = (self._agents[ag_num + 1],)
