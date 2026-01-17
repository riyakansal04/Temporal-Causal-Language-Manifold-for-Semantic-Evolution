from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import networkx as nx

from ..config import Paths
from ..causal.influence import CausalResults


def plot_trajectory(traj: pd.DataFrame, forecast: List[float], concept: str) -> str:
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(traj["time_bin"], traj["value"], label="observed")

    if len(forecast) and len(traj) > 0:
        # Try to infer frequency, fallback to 'M' (monthly) if not possible
        try:
            freq = pd.infer_freq(traj["time_bin"]) if len(traj) >= 3 else None
        except (ValueError, TypeError):
            freq = None
        
        if freq is None:
            # If can't infer, calculate average time delta
            if len(traj) >= 2:
                avg_delta = (traj["time_bin"].iloc[-1] - traj["time_bin"].iloc[0]) / (len(traj) - 1)
                future_idx = [traj["time_bin"].iloc[-1] + avg_delta * (i + 1) for i in range(len(forecast))]
            else:
                # Default to monthly offset if only one point
                future_idx = pd.date_range(traj["time_bin"].iloc[-1], periods=len(forecast)+1, freq='M', inclusive="right")
        else:
            future_idx = pd.date_range(traj["time_bin"].iloc[-1], periods=len(forecast)+1, freq=freq, inclusive="right")
        
        ax.plot(future_idx, forecast, label="forecast", linestyle='--')

    ax.set_title(f"Concept trajectory: {concept}")
    ax.set_xlabel("time")
    ax.set_ylabel("similarity")
    ax.legend()
    Paths.plots.mkdir(parents=True, exist_ok=True)
    out = Paths.plots / f"trajectory_{concept.replace(' ', '_')}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return str(out)


def plot_causal_graph(
    causal_results: CausalResults,
    significance_level: float = 0.10,
    min_edge_weight: float = 0.0,
    layout: str = "spring"
) -> str:
    """
    Visualize the causal graph from VAR + Granger causality results.
    
    Args:
        causal_results: Results from estimate_causal_matrix
        significance_level: P-value threshold for showing edges
        min_edge_weight: Minimum edge weight to display (1 - p_value)
        layout: Graph layout algorithm ('spring', 'circular', 'kamada_kawai', etc.)
        
    Returns:
        Path to saved plot file
    """
    import matplotlib.patches as mpatches
    
    # Build directed graph
    G = nx.DiGraph()
    sources = causal_results.sources
    p_matrix = causal_results.p_value_matrix
    
    # Add nodes
    for source in sources:
        G.add_node(source)
    
    # Add edges for significant relationships
    edge_weights = []
    edge_colors = []
    edge_labels = {}
    
    for i, cause in enumerate(sources):
        for j, effect in enumerate(sources):
            if i == j:
                continue
            
            p_val = p_matrix[i, j]
            if p_val < significance_level:
                weight = 1 - p_val  # Higher weight = more significant
                if weight >= min_edge_weight:
                    G.add_edge(cause, effect, weight=weight, p_value=p_val)
                    edge_weights.append(weight)
                    # Color by significance: darker = more significant
                    edge_colors.append(1 - p_val)
                    edge_labels[(cause, effect)] = f"{p_val:.3f}"
    
    if len(G.edges) == 0:
        # No significant relationships found
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.text(0.5, 0.5, 'No significant causal relationships found\n(p-value < {})'.format(significance_level),
                ha='center', va='center', fontsize=14, transform=ax.transAxes)
        ax.set_title('Causal Graph (VAR + Granger Causality)', fontsize=16, fontweight='bold')
        ax.axis('off')
    else:
        # Choose layout
        if layout == "spring":
            pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
        elif layout == "circular":
            pos = nx.circular_layout(G)
        elif layout == "kamada_kawai":
            try:
                pos = nx.kamada_kawai_layout(G)
            except:
                pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
        else:
            pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Draw nodes
        node_sizes = [2000] * len(G.nodes())
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                              node_size=node_sizes, ax=ax, alpha=0.9,
                              edgecolors='black', linewidths=2)
        
        # Draw edges with width and color based on significance
        if edge_weights:
            widths = [w * 3 for w in edge_weights]  # Scale for visibility
            edges = nx.draw_networkx_edges(G, pos, width=widths,
                                          edge_color=edge_colors,
                                          edge_cmap=plt.cm.Reds,
                                          alpha=0.7, ax=ax, arrows=True,
                                          arrowsize=20, arrowstyle='->',
                                          connectionstyle='arc3,rad=0.1')
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold', ax=ax)
        
        # Draw edge labels (p-values)
        if len(edge_labels) <= 20:  # Only show if not too many edges
            nx.draw_networkx_edge_labels(G, pos, edge_labels, 
                                        font_size=8, ax=ax, 
                                        bbox=dict(boxstyle='round,pad=0.3', 
                                                 facecolor='white', alpha=0.7))
        
        ax.set_title('Causal Graph (VAR + Granger Causality)', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.axis('off')
        
        # Add legend
        legend_elements = [
            mpatches.Patch(color='red', alpha=0.7, label='Significant causal relationship'),
            plt.Line2D([0], [0], color='black', linewidth=2, label='Edge width = significance strength'),
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    # Save figure
    Paths.plots.mkdir(parents=True, exist_ok=True)
    out = Paths.plots / "causal_graph.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return str(out)


