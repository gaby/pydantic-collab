"""Visualization utilities for agent topology graphs.

This module contains implementation details for rendering agent network topologies
using matplotlib and networkx. It is lazily imported only when visualization is needed.
"""
from collections.abc import Collection, Sequence
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from ._types import CollabAgent

if TYPE_CHECKING:
    import matplotlib.figure


# Color constants
TOOL_EDGE_COLOR = '#2E86AB'
HANDOFF_EDGE_COLOR = '#C73E1D'
NODE_COLOR_START = '#90EE90'
NODE_COLOR_FINAL = '#FFB6C1'
NODE_COLOR_BOTH = '#9370DB'
NODE_COLOR_OTHER = '#87CEEB'


def render_topology(
    agents: Sequence[CollabAgent],
    agent_tools: dict[CollabAgent, Collection[CollabAgent]],
    handoffs: dict[CollabAgent, Collection[CollabAgent]],
    starting_agent: CollabAgent,
    final_agent: CollabAgent,
    collab_name: str | None,
    figsize: tuple[float, float] = (12, 8),
    show: bool = True,
    save_path: str | None = None,
    title: str | None = None,
) -> 'matplotlib.figure.Figure':
    """Render a visual graph of the agent topology.

    Shows which agents can call which others as tools (dashed blue arrows)
    and handoffs (solid red arrows). Starting agent is highlighted in green,
    final agent in pink/red.

    Args:
        agents: List of all agents in the collab.
        agent_tools: Mapping of agents to their callable tool targets.
        handoffs: Mapping of agents to their handoff targets.
        starting_agent: The agent that starts execution.
        final_agent: The agent that can finalize output.
        collab_name: Name of the collab for default title.
        figsize: Figure size as (width, height) in inches.
        show: Whether to display the figure immediately in a window.
        save_path: Optional path to save the figure (e.g., 'topology.png').
        title: Optional title for the graph. Defaults to Collab name or 'Agent Topology'.

    Returns:
        matplotlib.figure.Figure: The generated figure object.

    Raises:
        ImportError: If networkx or matplotlib are not installed.
    """
    try:
        import matplotlib.patches as mpatches
        import matplotlib.pyplot as plt
        import networkx as nx
        from matplotlib.lines import Line2D
    except ImportError as e:
        raise ImportError(
            "Visualization requires 'networkx' and 'matplotlib'. "
            "Install with: pip install pydantic_collab[viz] or uv add \"pydantic_collab[viz]\"."
        ) from e

    # Collect edges
    tool_edges = [(a.name, t.name) for a, targets in agent_tools.items() for t in targets]
    handoff_edges = [(a.name, t.name) for a, targets in handoffs.items() for t in targets]

    # Build graph and get layout
    G = nx.DiGraph()
    G.add_nodes_from(a.name for a in agents)
    G.add_edges_from(tool_edges + handoff_edges)

    fig, ax = plt.subplots(figsize=figsize, facecolor='white')
    if not agents:
        ax.text(0.5, 0.5, 'No agents in collab', ha='center', va='center',
               transform=ax.transAxes, fontsize=14)
        ax.axis('off')
        return fig

    pos = _compute_layout(G, agents)
    positions_array = np.array(list(pos.values()))
    node_radius, font_size = _compute_node_sizing(positions_array, len(agents))

    # Draw edges
    tool_set, handoff_set = set(tool_edges), set(handoff_edges)
    bidirectional_tools = {tuple(sorted((u, v))) for u, v in tool_edges if (v, u) in tool_set}
    bidirectional_handoffs = {tuple(sorted((u, v))) for u, v in handoff_edges if (v, u) in handoff_set}
    offset_amount = node_radius * 0.3

    def draw_arrow(u: str, v: str, color: str, linestyle: str, lw: float, bidirectional: bool, offset: float):
        start, end = np.array(pos[u]), np.array(pos[v])
        direction = (end - start) / (np.linalg.norm(end - start) or 1)
        perp = np.array([-direction[1], direction[0]])
        start_pt = start + perp * offset + direction * (node_radius + 0.02)
        end_pt = end + perp * offset - direction * (node_radius + 0.05)
        ax.annotate('', xy=tuple(end_pt), xytext=tuple(start_pt),
                   arrowprops=dict(arrowstyle='<->' if bidirectional else '->', color=color,
                                  lw=lw, linestyle=linestyle, mutation_scale=20, alpha=0.8), zorder=1)

    drawn_tools: set[tuple[str, str]] = set()
    drawn_handoffs: set[tuple[str, str]] = set()
    for u, v in tool_edges:
        key = tuple(sorted((u, v)))
        if key in bidirectional_tools:
            if key not in drawn_tools:
                draw_arrow(u, v, TOOL_EDGE_COLOR, '--', 2, True, 0)
                drawn_tools.add(cast(tuple[str, str], key))
        else:
            offset = offset_amount if (v, u) in handoff_set else 0
            draw_arrow(u, v, TOOL_EDGE_COLOR, '--', 2, False, offset)

    for u, v in handoff_edges:
        key = tuple(sorted((u, v)))
        if key in bidirectional_handoffs:
            if key not in drawn_handoffs:
                offset = -offset_amount if (u, v) in tool_set or (v, u) in tool_set else 0
                draw_arrow(u, v, HANDOFF_EDGE_COLOR, '-', 3, True, offset)
                drawn_handoffs.add(cast(tuple[str, str], key))
        else:
            offset = -offset_amount if (u, v) in tool_set else 0
            draw_arrow(u, v, HANDOFF_EDGE_COLOR, '-', 3, False, offset)

    # Draw nodes
    node_colors = {
        (True, True): NODE_COLOR_BOTH, (True, False): NODE_COLOR_START,
        (False, True): NODE_COLOR_FINAL, (False, False): NODE_COLOR_OTHER
    }
    for agent in agents:
        x, y = pos[agent.name]
        is_start = agent == starting_agent
        is_final = agent == final_agent
        color = node_colors[(is_start, is_final)]
        ax.add_patch(mpatches.Circle((x, y), node_radius, facecolor=color, edgecolor='black',
                                    linewidth=2.5, zorder=10, alpha=0.9))
        ax.text(x, y, agent.name, ha='center', va='center', fontsize=font_size, fontweight='bold',
               zorder=11, bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='none', alpha=0.7))

    # Legend and finalize
    ax.legend(handles=[
        mpatches.Patch(facecolor=NODE_COLOR_START, edgecolor='black', linewidth=1.5, label='Start'),
        mpatches.Patch(facecolor=NODE_COLOR_FINAL, edgecolor='black', linewidth=1.5, label='Final'),
        mpatches.Patch(facecolor=NODE_COLOR_BOTH, edgecolor='black', linewidth=1.5, label='Start & Final'),
        mpatches.Patch(facecolor=NODE_COLOR_OTHER, edgecolor='black', linewidth=1.5, label='Other'),
        Line2D([0], [0], color=TOOL_EDGE_COLOR, linestyle='--', lw=2.5, label='Tool Call (→ or ↔)'),
        Line2D([0], [0], color=HANDOFF_EDGE_COLOR, linestyle='-', lw=3, label='Handoff (→ or ↔)'),
    ], loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=10, framealpha=0.95, edgecolor='gray', fancybox=True)

    if len(positions_array) > 0:
        x_min, x_max = positions_array[:, 0].min(), positions_array[:, 0].max()
        y_min, y_max = positions_array[:, 1].min(), positions_array[:, 1].max()
        pad_x, pad_y = max(0.5, (x_max - x_min) * 0.2), max(0.5, (y_max - y_min) * 0.2)
        ax.set_xlim(x_min - pad_x, x_max + pad_x)
        ax.set_ylim(y_min - pad_y, y_max + pad_y)
    else:
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)

    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title or collab_name or 'Agent Topology', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
    
    # Automatically show the plot in a window (like matplotlib's .plot())
    if show:
        plt.show()
    
    return fig


def _compute_layout(G: Any, agents: Sequence['CollabAgent']) -> dict[str, Any]:
    """Compute node positions using best available layout algorithm."""
    import networkx as nx
    
    num_nodes = len(agents)
    k = max(1.5, 3.0 / np.sqrt(num_nodes))
    
    for layout_func in [
        lambda: nx.spring_layout(G, k=k, iterations=200, seed=42),
        lambda: nx.spring_layout(G, k=k*1.2, iterations=300, seed=42),
        lambda: nx.circular_layout(G),
    ]:
        try:
            pos = layout_func()
            positions = np.array(list(pos.values()))
            if len(positions) > 1:
                min_dist = np.min([np.linalg.norm(positions[i] - positions[j]) 
                                 for i in range(len(positions)) 
                                 for j in range(i+1, len(positions))])
                if min_dist > 0.3:
                    return pos
        except Exception:
            continue
    
    return nx.spring_layout(G, k=k, iterations=200, seed=42)


def _compute_node_sizing(positions_array: Any, num_nodes: int) -> tuple[float, int]:
    """Calculate adaptive node radius and font size."""        
    if len(positions_array) > 1:
        max_range = max(
            positions_array[:, 0].max() - positions_array[:, 0].min(),
            positions_array[:, 1].max() - positions_array[:, 1].min(),
            1.0
        )
        node_radius = max(0.15, min(0.4, max_range * 0.08))
    else:
        node_radius = 0.25
    
    font_size = max(8, min(12, int(14 - num_nodes * 0.5)))
    return node_radius, font_size
