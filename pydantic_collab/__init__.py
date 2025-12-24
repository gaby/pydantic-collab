"""Agent Graph - Multi-agent orchestration framework.

Provides high-level abstractions for building multi-agent systems with Pydantic AI.
"""

from ._types import (
    AgentContext,
    CollabAgent,
    CollabError,
    CollabRunResult,
    CollabSettings,
    HandOffBase,
    PromptBuilderContext,
)
from ._utils import default_build_agent_prompt
from .collab import (
    Collab,
    CollabState,
)
from .custom_collabs import ForwardHandoffCollab, MeshCollab, StarCollab

# Type alias for convenience (kept for backwards compatibility)
AgentOutputType = HandOffBase

__all__ = [
    # Core
    'AgentContext',
    'CollabAgent',
    # Output types
    'AgentOutputType',
    'HandOffBase',
    # Results
    'CollabRunResult',
    # Custom Collabs
    'ForwardHandoffCollab',
    'MeshCollab',
    'StarCollab',
    # Prompt/Context builders
    'PromptBuilderContext',
    'default_build_agent_prompt',
    # Settings
    'CollabSettings',
    # Collab
    'Collab',
    'CollabState',
    # Exceptions
    'CollabError',
]
