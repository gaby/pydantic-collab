"""Test that all __all__ exports can be imported successfully."""

import pytest



def test_all_exports_importable():
    """Test that all exports in __all__ are importable."""
    import pydantic_collab as agent_graph
    from pydantic_collab import __all__

    for name in __all__:
        assert hasattr(agent_graph, name), f"Export '{name}' not found in module"
        obj = getattr(agent_graph, name)
        assert obj is not None, f"Export '{name}' is None"


def test_core_classes():
    """Test that core classes can be imported."""
    from pydantic_collab import (
        Collab,
        CollabAgent,
        CollabError,
        CollabState,
        PromptBuilderContext,
    )
    from pydantic_collab._types import AgentContext, HandOffBase
    from pydantic_collab._utils import default_build_agent_prompt

    # Verify they are actual classes/types/functions
    assert AgentContext is not None
    assert CollabAgent is not None
    assert HandOffBase is not None
    assert Collab is not None
    assert PromptBuilderContext is not None
    assert CollabState is not None
    assert CollabError is not None
    assert callable(default_build_agent_prompt)




def test_handoff_output_model():
    """Test HandoffOutput model instantiation."""
    from pydantic_collab._types import HandOffBase

    output = HandOffBase[str](next_agent='Agent2', query='Hello')
    assert output.next_agent == 'Agent2'
    assert output.query == 'Hello'
    assert output.reasoning is None




def test_swarm_state():
    """Test CollabState dataclass."""
    from pydantic_collab import CollabState

    state = CollabState(query='test query')
    assert state.query == 'test query'
    assert state.final_output == ''
    assert state.agent_contexts == {}
    assert state.execution_path == []
    assert state.execution_history == []


def test_swarm_validation_error():
    """Test CollabError exception."""
    from pydantic_collab import CollabError

    with pytest.raises(CollabError):
        raise CollabError('test error')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
