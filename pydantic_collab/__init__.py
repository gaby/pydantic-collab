"""Agent Graph - Multi-agent orchestration framework.

Provides high-level abstractions for building multi-agent systems with Pydantic AI.
"""

__version__ = '0.1.0'
from ._types import (
    CollabAgent,
    CollabError,
    CollabSettings,
    PromptBuilderContext,
)
from .collab import (
    Collab,
    CollabState,
)
from .custom_collabs import ForwardHandoffCollab, MeshCollab, StarCollab

__all__ = [
    # Core
    'CollabAgent',
    'Collab',
    'CollabState',
    # Custom Collabs
    'ForwardHandoffCollab',
    'MeshCollab',
    'StarCollab',
    # Prompt/Context builders
    'PromptBuilderContext',
    # Settings
    'CollabSettings',
    # Exceptions
    'CollabError',
]
