import pytest
import tempfile
import os
from pathlib import Path

# Skip all tests in this module if viz dependencies are not installed
pytest.importorskip("matplotlib")
pytest.importorskip("networkx")

from pydantic_collab._types import CollabAgent

class MockAgent(CollabAgent):
    def __init__(self, name: str):
        super().__init__(name=name)

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return isinstance(other, MockAgent) and self.name == other.name

def test_render_topology_smoke():
    """Basic smoke test to ensure render_topology runs without crashing."""
    pytest.importorskip("matplotlib")
    pytest.importorskip("networkx")

    from pydantic_collab._viz import render_topology

    agent1 = MockAgent(name="Agent1")
    agent2 = MockAgent(name="Agent2")
    agents = [agent1, agent2]
    agent_tools = {agent1: [agent2]}
    handoffs = {agent2: [agent1]}

    fig = render_topology(
        agents=agents,
        agent_tools=agent_tools,
        handoffs=handoffs,
        starting_agent=agent1,
        final_agent=agent2,
        collab_name="TestCollab",
        show=False,  # Don't show window in test
    )

    assert fig is not None
    import matplotlib.pyplot as plt
    plt.close(fig)  # Clean up

def test_render_topology_save_path():
    """Test that the figure is saved when a save_path is provided."""
    pytest.importorskip("matplotlib")
    pytest.importorskip("networkx")

    from pydantic_collab._viz import render_topology
    import matplotlib.pyplot as plt

    agent1 = MockAgent(name="Agent1")
    agents = [agent1]

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "test.png")

        fig = render_topology(
            agents=agents,
            agent_tools={},
            handoffs={},
            starting_agent=agent1,
            final_agent=agent1,
            collab_name="TestCollab",
            show=False,
            save_path=save_path,
        )

        assert fig is not None
        assert os.path.exists(save_path), "Figure was not saved to file"

        # Verify it's a valid PNG file
        assert Path(save_path).stat().st_size > 0, "Saved file is empty"
        plt.close(fig)

def test_render_topology_no_agents():
    """Test that rendering a topology with no agents works correctly."""
    pytest.importorskip("matplotlib")
    pytest.importorskip("networkx")

    from pydantic_collab._viz import render_topology
    import matplotlib.pyplot as plt

    fig = render_topology(
        agents=[],
        agent_tools={},
        handoffs={},
        starting_agent=None,
        final_agent=None,
        collab_name="EmptyCollab",
        show=False,
    )

    assert fig is not None
    plt.close(fig)

def test_render_topology_title_logic():
    """Test various combinations of title and collab_name."""
    pytest.importorskip("matplotlib")
    pytest.importorskip("networkx")

    from pydantic_collab._viz import render_topology
    import matplotlib.pyplot as plt

    agent1 = MockAgent(name="Agent1")
    agents = [agent1]

    # Case 1: Custom title provided
    fig1 = render_topology(
        agents=agents, agent_tools={}, handoffs={}, starting_agent=agent1, final_agent=agent1,
        collab_name="CollabName", title="Custom Title", show=False
    )
    assert fig1.axes[0].get_title() == "Custom Title"
    plt.close(fig1)

    # Case 2: Only collab_name provided
    fig2 = render_topology(
        agents=agents, agent_tools={}, handoffs={}, starting_agent=agent1, final_agent=agent1,
        collab_name="CollabName Only", title=None, show=False
    )
    assert fig2.axes[0].get_title() == "CollabName Only"
    plt.close(fig2)

    # Case 3: Neither provided
    fig3 = render_topology(
        agents=agents, agent_tools={}, handoffs={}, starting_agent=agent1, final_agent=agent1,
        collab_name=None, title=None, show=False
    )
    assert fig3.axes[0].get_title() == "Agent Topology"
    plt.close(fig3)

def test_render_topology_bidirectional_edges_and_offsets():
    """Test bidirectional edges and edges with offsets."""
    pytest.importorskip("matplotlib")
    pytest.importorskip("networkx")

    from pydantic_collab._viz import render_topology
    import matplotlib.pyplot as plt

    agent1 = MockAgent(name="Agent1")
    agent2 = MockAgent(name="Agent2")
    agent3 = MockAgent(name="Agent3")
    agents = [agent1, agent2, agent3]

    # Test bidirectional tool edge
    fig1 = render_topology(
        agents=agents,
        agent_tools={agent1: [agent2], agent2: [agent1]},
        handoffs={},
        starting_agent=agent1,
        final_agent=agent2,
        collab_name="BidirectionalTools",
        show=False,
    )
    assert fig1 is not None
    plt.close(fig1)

    # Test bidirectional handoff edge
    fig2 = render_topology(
        agents=agents,
        agent_tools={},
        handoffs={agent1: [agent2], agent2: [agent1]},
        starting_agent=agent1,
        final_agent=agent2,
        collab_name="BidirectionalHandoffs",
        show=False,
    )
    assert fig2 is not None
    plt.close(fig2)

    # Test overlapping edges (tool A->B, handoff B->A)
    fig3 = render_topology(
        agents=agents,
        agent_tools={agent1: [agent2]},
        handoffs={agent2: [agent1]},
        starting_agent=agent1,
        final_agent=agent2,
        collab_name="OverlappingEdges",
        show=False,
    )
    assert fig3 is not None
    plt.close(fig3)
def test_compute_layout_logic():
    """Test _compute_layout with various agent configurations."""
    pytest.importorskip("matplotlib")
    networkx = pytest.importorskip("networkx")

    from pydantic_collab._viz import _compute_layout
    import numpy as np

    # Test with 2 agents
    agent1 = MockAgent(name="Agent1")
    agent2 = MockAgent(name="Agent2")
    agents = [agent1, agent2]
    G = networkx.DiGraph()
    G.add_nodes_from(a.name for a in agents)

    pos = _compute_layout(G, agents)
    assert len(pos) == 2
    assert 'Agent1' in pos
    assert 'Agent2' in pos

    # Verify positions are numpy arrays or tuples
    assert hasattr(pos['Agent1'], '__iter__')
    assert hasattr(pos['Agent2'], '__iter__')

    # Test with single agent
    agent_single = MockAgent(name="Single")
    G_single = networkx.DiGraph()
    G_single.add_node(agent_single.name)
    pos_single = _compute_layout(G_single, [agent_single])
    assert len(pos_single) == 1
    assert 'Single' in pos_single

    # Test with many agents
    many_agents = [MockAgent(name=f"Agent{i}") for i in range(10)]
    G_many = networkx.DiGraph()
    G_many.add_nodes_from(a.name for a in many_agents)
    pos_many = _compute_layout(G_many, many_agents)
    assert len(pos_many) == 10


def test_compute_node_sizing():
    """Test _compute_node_sizing for different numbers of nodes."""
    pytest.importorskip("matplotlib")
    pytest.importorskip("networkx")
    numpy = pytest.importorskip("numpy")

    from pydantic_collab._viz import _compute_node_sizing

    # Test with typical positions
    positions = numpy.array([[0, 0], [1, 1], [2, 0]])
    radius, font_size = _compute_node_sizing(positions, 3)

    assert isinstance(radius, float)
    assert isinstance(font_size, int)
    assert 0.15 <= radius <= 0.4
    assert 8 <= font_size <= 12

    # Test with single position
    single_pos = numpy.array([[0, 0]])
    radius_single, font_size_single = _compute_node_sizing(single_pos, 1)
    assert radius_single == 0.25  # Default for single node
    assert 8 <= font_size_single <= 14

    # Test with many nodes (font size should decrease)
    _, font_size_few = _compute_node_sizing(numpy.array([[0, 0], [1, 1]]), 2)
    _, font_size_many = _compute_node_sizing(numpy.array([[i, 0] for i in range(20)]), 20)
    assert font_size_many <= font_size_few


def test_render_topology_with_show():
    """Test render_topology with show=True to cover plt.show() call."""
    pytest.importorskip("matplotlib")
    pytest.importorskip("networkx")

    from pydantic_collab._viz import render_topology
    import matplotlib.pyplot as plt
    from unittest.mock import patch

    agent1 = MockAgent(name="Agent1")
    agents = [agent1]

    # Mock plt.show() to prevent actual window display
    with patch.object(plt, 'show') as mock_show:
        fig = render_topology(
            agents=agents,
            agent_tools={},
            handoffs={},
            starting_agent=agent1,
            final_agent=agent1,
            collab_name="TestCollab",
            show=True,  # This should call plt.show()
        )

        # Verify show was called
        mock_show.assert_called_once()
        plt.close(fig)


def test_render_topology_import_error():
    """Test that ImportError is raised when matplotlib/networkx are not available."""
    from unittest.mock import patch
    import sys

    # Temporarily hide matplotlib and networkx
    with patch.dict(sys.modules, {'matplotlib': None, 'networkx': None}):
        # Force reimport to trigger ImportError
        import importlib
        import pydantic_collab._viz
        importlib.reload(pydantic_collab._viz)

        from pydantic_collab._viz import render_topology

        agent1 = MockAgent(name="Agent1")
        agents = [agent1]

        with pytest.raises(ImportError, match="Visualization requires"):
            render_topology(
                agents=agents,
                agent_tools={},
                handoffs={},
                starting_agent=agent1,
                final_agent=agent1,
                collab_name="TestCollab",
                show=False,
            )

        # Restore the module
        importlib.reload(pydantic_collab._viz)


def test_compute_layout_fallback_to_circular():
    """Test _compute_layout falls back to circular when spring layouts fail."""
    pytest.importorskip("matplotlib")
    networkx = pytest.importorskip("networkx")
    numpy = pytest.importorskip("numpy")
    from unittest.mock import patch, MagicMock

    from pydantic_collab._viz import _compute_layout

    # Create a simple graph
    agent1 = MockAgent(name="Agent1")
    agent2 = MockAgent(name="Agent2")
    agents = [agent1, agent2]
    G = networkx.DiGraph()
    G.add_nodes_from(a.name for a in agents)

    # Mock spring_layout to return positions too close together
    def mock_spring_layout_close(*args, **kwargs):
        return {'Agent1': numpy.array([0.0, 0.0]), 'Agent2': numpy.array([0.1, 0.1])}

    with patch.object(networkx, 'spring_layout', mock_spring_layout_close):
        pos = _compute_layout(G, agents)
        # Should still return valid positions even if spring layout returns close positions
        assert len(pos) == 2


def test_compute_layout_exception_handling():
    """Test _compute_layout handles exceptions in layout functions."""
    pytest.importorskip("matplotlib")
    networkx = pytest.importorskip("networkx")
    from unittest.mock import patch

    from pydantic_collab._viz import _compute_layout

    agent1 = MockAgent(name="Agent1")
    agent2 = MockAgent(name="Agent2")
    agents = [agent1, agent2]
    G = networkx.DiGraph()
    G.add_nodes_from(a.name for a in agents)

    call_count = [0]
    original_spring = networkx.spring_layout
    original_circular = networkx.circular_layout

    def mock_spring_first_fail(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] <= 2:
            raise Exception("Spring layout failed")
        return original_spring(*args, **kwargs)

    with patch.object(networkx, 'spring_layout', mock_spring_first_fail):
        with patch.object(networkx, 'circular_layout', original_circular):
            pos = _compute_layout(G, agents)
            # Should still return valid positions via fallback
            assert len(pos) == 2


def test_render_topology_empty_agents():
    """Test render_topology with empty agents list uses default axis limits."""
    plt = pytest.importorskip("matplotlib.pyplot")
    pytest.importorskip("networkx")

    from pydantic_collab._viz import render_topology

    fig = render_topology(
        agents=[],
        agent_tools={},
        handoffs={},
        starting_agent=None,
        final_agent=None,
        collab_name="EmptyCollab",
        show=False,
    )

    # Should return a valid figure even with empty agents (uses default axis limits -1,1)
    assert fig is not None
    plt.close(fig)
