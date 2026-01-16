"""Pydantic Collab - a Multi-agent orchestration framework built on top of Pydantic-AI."""

__version__ = '0.2.0'

from ._types import (
    AgentMemory,
    CollabAgent,
    CollabError,
    CollabSettings,
    PromptBuilderContext,
)
from .collab import (
    Collab,
    CollabState,
)
from .custom_collabs import MeshCollab, PipelineCollab, StarCollab

__all__ = [
    # Core
    'AgentMemory',
    'CollabAgent',
    'Collab',
    'CollabState',
    # Custom Collabs
    'PipelineCollab',
    'MeshCollab',
    'StarCollab',
    # Prompt/Context builders
    'PromptBuilderContext',
    # Settings
    'CollabSettings',
    # Exceptions
    'CollabError',
]
