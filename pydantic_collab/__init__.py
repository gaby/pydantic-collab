"""Pydantic Collab - a Multi-agent orchestration framework built on top of Pydantic-AI."""

__version__ = '0.1.1'

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
from .custom_collabs import MeshCollab, PipelineCollab, StarCollab

__all__ = [
    # Core
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
