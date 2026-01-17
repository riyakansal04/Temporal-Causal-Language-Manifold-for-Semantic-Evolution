"""
Generate Concept Trajectory Visualizations for AG News Dataset
Creates temporal evolution plots for different concepts across news categories
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pathlib import Path

from tclm.semantic.embed import embed_corpus
from tclm.semantic.manifold import build_concept_trajectory
from tclm.forecast.models import forecast_trajectory


def load_ag_news(csv_path, sample_size=5000):
    """Load and prepare AG News data"""
    print(f"\n[Loading AG News from {csv_path}]")
    
    df = pd.read_csv(csv_path)
    df.columns = ['class', 'title', 'description']
    
    class_map = {1: 'World', 2: 'Sports', 3: 'Business', 4: 'Sci/Tech'}
    df['category'] = df['class'].map(class_map)
    df['text'] = df['title'].fillna('').astype(str) + '. ' + df['description'].fillna('').astype(str)
    
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)
    
    print(f"  ✓ Loaded {len(df)} samples")
    return df


def add_temporal_structure(df, months=24):
    """Add realistic temporal structure"""
    print("\n[Creating Temporal Structure]")
    
    start_date = datetime(2022, 1, 1)
    n_bins = months * 4  # Weekly bins
    
    # Assign dates
    dates = []
    for i in range(len(df)):
        bin_idx = i % n_bins
        date = start_date + timedelta(days=7 * bin_idx)
        dates.append(date)
    
    df['published'] = dates
    df['source'] = df['category'].apply(lambda x: x.lower().replace('/', '_'))
    df['time_bin'] = pd.to_datetime(df['published']).dt.to_period('W').dt.to_timestamp()
    
    print(f"  ✓ Created {df['time_bin'].nunique()} weekly time bins")
    print(f"  ✓ Date range: {df['time_bin'].min()} to {df['time_bin'].max()}")
    
    return df


def plot_concept_trajectories(embedded_df, concepts, output_dir='outputs/plots'):
    """Generate trajectory plots for multiple concepts"""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("GENERATING CONCEPT TRAJECTORY PLOTS")
    print("="*70)
    
    for concept in concepts:
        print(f"\n[Concept: '{concept}']")
        
        # Build trajectory
        traj = build_concept_trajectory(embedded_df, concept=concept)
        print(f"  ✓ Built trajectory with {len(traj)} time points")
        
        # Forecast
        fcst = forecast_trajectory(traj, steps=12)  # 12 weeks ahead
        print(f"  ✓ Generated {len(fcst)} forecast steps")
        
        # Calculate statistics
        prevalence = embedded_df['text'].str.lower().str.contains(concept.lower()).mean()
        mean_sim = traj['value'].mean()
        trend = np.polyfit(range(len(traj)), traj['value'], 1)[0]
        
        print(f"  • Document prevalence: {prevalence:.2%}")
        print(f"  • Mean similarity: {mean_sim:.4f}")
        print(f"  • Trend: {'↑ increasing' if trend > 0 else '↓ decreasing'} ({trend:.6f})")
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot observed trajectory
        ax.plot(traj['time_bin'], traj['value'], 
                marker='o', linewidth=2, markersize=4,
                label='Observed Trajectory', color='#2E86AB')
        
        # Plot forecast
        if len(fcst) > 0 and len(traj) >= 3:
            try:
                # Calculate time delta
                if len(traj) >= 2:
                    avg_delta = (traj['time_bin'].iloc[-1] - traj['time_bin'].iloc[0]) / (len(traj) - 1)
                    future_dates = [traj['time_bin'].iloc[-1] + avg_delta * (i + 1) for i in range(len(fcst))]
                else:
                    future_dates = pd.date_range(traj['time_bin'].iloc[-1], periods=len(fcst)+1, freq='W')[1:]
                
                ax.plot(future_dates, fcst,
                       marker='s', linewidth=2, markersize=4, linestyle='--',
                       label='Forecast (12 weeks)', color='#A23B72', alpha=0.8)
                
                # Shade forecast region
                ax.axvspan(traj['time_bin'].iloc[-1], future_dates[-1], 
                          alpha=0.1, color='gray', label='Forecast Period')
            except Exception as e:
                print(f"  ⚠ Could not plot forecast: {e}")
        
        # Styling
        ax.set_title(f'Temporal Trajectory: "{concept}" in News Articles', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Semantic Similarity Score', fontsize=12)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='best', framealpha=0.9)
        
        # Add statistics box
        stats_text = f"Statistics:\n"
        stats_text += f"• Prevalence: {prevalence:.1%}\n"
        stats_text += f"• Mean Similarity: {mean_sim:.3f}\n"
        stats_text += f"• Trend: {'↑' if trend > 0 else '↓'} {abs(trend):.6f}"
        
        ax.text(0.02, 0.98, stats_text,
               transform=ax.transAxes,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
               fontsize=9, family='monospace')
        
        # Rotate x-axis labels
        plt.xticks(rotation=45, ha='right')
        
        fig.tight_layout()
        
        # Save
        filename = f"trajectory_{concept.replace(' ', '_').replace('/', '_')}.png"
        filepath = output_dir / filename
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"  ✓ Saved: {filepath}")
    
    print(f"\n{'='*70}")
    print(f"All trajectory plots saved to: {output_dir}")
    print(f"{'='*70}\n")


def plot_category_comparison(embedded_df, output_dir='outputs/plots'):
    """Compare semantic trajectories across news categories"""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n[Creating Category Comparison Plot]")
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    categories = embedded_df['category'].unique()
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#06A77D']
    
    for i, category in enumerate(categories):
        # Filter by category
        cat_df = embedded_df[embedded_df['category'] == category].copy()
        
        # Build trajectory (using category name as concept)
        traj = build_concept_trajectory(cat_df, concept=category.lower())
        
        # Plot
        ax.plot(traj['time_bin'], traj['value'],
               marker='o', linewidth=2, markersize=3,
               label=category, color=colors[i % len(colors)])
    
    ax.set_title('Semantic Trajectory Comparison Across News Categories', 
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Within-Category Semantic Coherence', fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', framealpha=0.9, fontsize=11)
    
    plt.xticks(rotation=45, ha='right')
    fig.tight_layout()
    
    filepath = output_dir / 'category_comparison.png'
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"  ✓ Saved: {filepath}\n")


def main():
    import sys
    
    print("\n" + "="*70)
    print("AG NEWS CONCEPT TRAJECTORY VISUALIZATION")
    print("="*70)
    
    # Load data
    csv_path = sys.argv[1] if len(sys.argv) > 1 else 'data/train.csv'
    df = load_ag_news(csv_path, sample_size=5000)
    
    # Add temporal structure
    df = add_temporal_structure(df, months=24)
    
    # Generate embeddings
    print("\n[Generating Embeddings...]")
    embedded_df = embed_corpus(df)
    print(f"  ✓ Generated {len(embedded_df)} embeddings")
    
    # Concepts to track
    concepts = [
        'technology',
        'economy',
        'politics',
        'sports',
        'war',
        'election',
        'market',
        'player'
    ]
    
    # Generate trajectory plots
    plot_concept_trajectories(embedded_df, concepts)
    
    # Generate category comparison
    plot_category_comparison(embedded_df)
    
    print("\n" + "="*70)
    print("VISUALIZATION COMPLETE")
    print("="*70)
    print(f"\nView plots in: outputs/plots/")
    print(f"  • Individual concepts: trajectory_<concept>.png")
    print(f"  • Category comparison: category_comparison.png")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
