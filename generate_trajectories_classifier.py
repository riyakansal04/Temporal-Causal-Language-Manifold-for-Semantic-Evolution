"""
Generate Classification-Based Trajectory Visualizations for AG News Dataset
Uses trained classification models to track category distribution evolution over time
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns


def load_ag_news(csv_path, sample_size=10000):
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
    print(f"  • Category distribution: {df['category'].value_counts().to_dict()}")
    return df


def add_temporal_structure(df, months=24):
    """Add realistic temporal structure"""
    print("\n[Creating Temporal Structure]")
    
    start_date = datetime(2022, 1, 1)
    n_bins = months * 4  # Weekly bins
    
    # Assign dates (cyclic pattern to ensure all categories appear in each bin)
    dates = []
    for i in range(len(df)):
        bin_idx = i % n_bins
        date = start_date + timedelta(days=7 * bin_idx)
        dates.append(date)
    
    df['published'] = dates
    df['time_bin'] = pd.to_datetime(df['published']).dt.to_period('W').dt.to_timestamp()
    
    print(f"  ✓ Created {df['time_bin'].nunique()} weekly time bins")
    print(f"  ✓ Date range: {df['time_bin'].min()} to {df['time_bin'].max()}")
    
    return df


def train_classifier(df):
    """Train Logistic Regression classifier with TF-IDF"""
    print("\n[Training Classification Model]")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], df['category'], test_size=0.2, random_state=42, stratify=df['category']
    )
    
    print(f"  • Training samples: {len(X_train)}")
    print(f"  • Test samples: {len(X_test)}")
    
    # TF-IDF vectorization
    print("  • Creating TF-IDF features...")
    vectorizer = TfidfVectorizer(max_features=10000, stop_words='english', ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    print(f"  • Feature shape: {X_train_tfidf.shape}")
    
    # Train classifier
    print("  • Training Logistic Regression...")
    classifier = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    classifier.fit(X_train_tfidf, y_train)
    
    # Evaluate
    train_acc = classifier.score(X_train_tfidf, y_train)
    test_acc = classifier.score(X_test_tfidf, y_test)
    
    print(f"  ✓ Training accuracy: {train_acc:.2%}")
    print(f"  ✓ Test accuracy: {test_acc:.2%}")
    
    return classifier, vectorizer


def generate_category_probabilities(df, classifier, vectorizer):
    """Generate category probability predictions for all documents"""
    print("\n[Generating Category Predictions]")
    
    # Transform text to TF-IDF
    X_tfidf = vectorizer.transform(df['text'])
    
    # Get probability predictions
    probabilities = classifier.predict_proba(X_tfidf)
    
    # Add probabilities to dataframe
    categories = classifier.classes_
    for i, cat in enumerate(categories):
        df[f'prob_{cat}'] = probabilities[:, i]
    
    print(f"  ✓ Generated probabilities for {len(df)} documents")
    print(f"  • Categories: {list(categories)}")
    
    return df


def plot_category_distribution_over_time(df, output_dir='outputs/plots'):
    """Plot how category distribution changes over time"""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n[Creating Category Distribution Trajectory]")
    
    categories = ['World', 'Sports', 'Business', 'Sci/Tech']
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#06A77D']
    
    # Group by time bin and calculate average probability
    time_probs = df.groupby('time_bin')[[f'prob_{cat}' for cat in categories]].mean()
    
    # Plot stacked area chart
    fig, ax = plt.subplots(figsize=(14, 8))
    
    ax.stackplot(time_probs.index, 
                 *[time_probs[f'prob_{cat}'] for cat in categories],
                 labels=categories,
                 colors=colors,
                 alpha=0.8)
    
    ax.set_title('News Category Distribution Over Time (Classification-Based)', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Time', fontsize=13)
    ax.set_ylabel('Category Probability', fontsize=13)
    ax.legend(loc='upper left', framealpha=0.95, fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    plt.xticks(rotation=45, ha='right')
    fig.tight_layout()
    
    filepath = output_dir / 'category_distribution_trajectory.png'
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"  ✓ Saved: {filepath}")
    
    return time_probs


def plot_individual_category_trajectories(df, output_dir='outputs/plots'):
    """Plot individual trajectory for each category"""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n[Creating Individual Category Trajectories]")
    
    categories = ['World', 'Sports', 'Business', 'Sci/Tech']
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#06A77D']
    
    # Create 2x2 subplot
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()
    
    for i, (cat, color) in enumerate(zip(categories, colors)):
        ax = axes[i]
        
        # Group by time bin
        time_prob = df.groupby('time_bin')[f'prob_{cat}'].agg(['mean', 'std', 'count'])
        
        # Calculate trend
        x = np.arange(len(time_prob))
        trend_coef = np.polyfit(x, time_prob['mean'], 1)[0]
        trend_line = np.poly1d(np.polyfit(x, time_prob['mean'], 1))(x)
        
        # Calculate statistics
        overall_mean = time_prob['mean'].mean()
        max_prob = time_prob['mean'].max()
        min_prob = time_prob['mean'].min()
        
        # Plot trajectory with confidence interval
        ax.plot(time_prob.index, time_prob['mean'], 
               marker='o', linewidth=2.5, markersize=5,
               label=f'{cat} Probability', color=color)
        
        # Add trend line
        ax.plot(time_prob.index, trend_line,
               linestyle='--', linewidth=2, alpha=0.6,
               label=f'Trend ({"↑" if trend_coef > 0 else "↓"} {abs(trend_coef):.4f})',
               color='red')
        
        # Add confidence band (±1 std)
        ax.fill_between(time_prob.index,
                        time_prob['mean'] - time_prob['std'],
                        time_prob['mean'] + time_prob['std'],
                        alpha=0.2, color=color)
        
        # Styling
        ax.set_title(f'{cat} News Prevalence Over Time', 
                    fontsize=13, fontweight='bold')
        ax.set_xlabel('Time', fontsize=11)
        ax.set_ylabel('Average Probability', fontsize=11)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='best', framealpha=0.9, fontsize=9)
        
        # Add statistics box
        stats_text = f"Statistics:\n"
        stats_text += f"• Mean: {overall_mean:.3f}\n"
        stats_text += f"• Range: [{min_prob:.3f}, {max_prob:.3f}]\n"
        stats_text += f"• Trend: {'↑' if trend_coef > 0 else '↓'} {abs(trend_coef):.5f}"
        
        ax.text(0.02, 0.98, stats_text,
               transform=ax.transAxes,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6),
               fontsize=8, family='monospace')
        
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    fig.suptitle('Category-Specific Trajectory Analysis (Classification-Based)', 
                fontsize=16, fontweight='bold', y=1.00)
    fig.tight_layout()
    
    filepath = output_dir / 'individual_category_trajectories.png'
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"  ✓ Saved: {filepath}")


def plot_category_dominance_heatmap(df, output_dir='outputs/plots'):
    """Create heatmap showing dominant category per time bin"""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n[Creating Category Dominance Heatmap]")
    
    categories = ['World', 'Sports', 'Business', 'Sci/Tech']
    
    # Group by time bin and get probabilities
    time_probs = df.groupby('time_bin')[[f'prob_{cat}' for cat in categories]].mean()
    time_probs.columns = categories
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(14, 10))
    
    sns.heatmap(time_probs.T, cmap='YlOrRd', annot=False, fmt='.3f',
                cbar_kws={'label': 'Average Probability'},
                linewidths=0.5, ax=ax)
    
    ax.set_title('Category Dominance Heatmap Over Time', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Time Bin', fontsize=13)
    ax.set_ylabel('News Category', fontsize=13)
    
    # Rotate x-axis labels
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=7)
    
    fig.tight_layout()
    
    filepath = output_dir / 'category_dominance_heatmap.png'
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"  ✓ Saved: {filepath}")


def generate_summary_report(df, time_probs):
    """Generate text summary of trajectory insights"""
    
    print("\n" + "="*70)
    print("CLASSIFICATION-BASED TRAJECTORY SUMMARY")
    print("="*70)
    
    categories = ['World', 'Sports', 'Business', 'Sci/Tech']
    
    for cat in categories:
        prob_col = f'prob_{cat}'
        
        # Calculate statistics
        mean_prob = df[prob_col].mean()
        time_series = time_probs[prob_col]
        
        # Calculate trend
        x = np.arange(len(time_series))
        trend_coef = np.polyfit(x, time_series, 1)[0]
        
        # Calculate volatility (std of differences)
        volatility = time_series.diff().std()
        
        # Find peak and trough
        peak_time = time_series.idxmax()
        peak_prob = time_series.max()
        trough_time = time_series.idxmin()
        trough_prob = time_series.min()
        
        print(f"\n[{cat}]")
        print(f"  • Average Probability: {mean_prob:.3f}")
        print(f"  • Trend: {'↑ Increasing' if trend_coef > 0 else '↓ Decreasing'} ({trend_coef:.5f}/week)")
        print(f"  • Volatility: {volatility:.4f}")
        print(f"  • Peak: {peak_prob:.3f} on {peak_time.date()}")
        print(f"  • Trough: {trough_prob:.3f} on {trough_time.date()}")
    
    print("\n" + "="*70)


def main():
    import sys
    
    print("\n" + "="*70)
    print("CLASSIFICATION-BASED TRAJECTORY VISUALIZATION")
    print("="*70)
    
    # Load data
    csv_path = sys.argv[1] if len(sys.argv) > 1 else 'data/train.csv'
    df = load_ag_news(csv_path, sample_size=10000)
    
    # Add temporal structure
    df = add_temporal_structure(df, months=24)
    
    # Train classifier
    classifier, vectorizer = train_classifier(df)
    
    # Generate predictions
    df = generate_category_probabilities(df, classifier, vectorizer)
    
    # Create visualizations
    time_probs = plot_category_distribution_over_time(df)
    plot_individual_category_trajectories(df)
    plot_category_dominance_heatmap(df)
    
    # Generate summary
    generate_summary_report(df, time_probs)
    
    print("\n" + "="*70)
    print("VISUALIZATION COMPLETE")
    print("="*70)
    print(f"\nView plots in: outputs/plots/")
    print(f"  • category_distribution_trajectory.png (stacked area)")
    print(f"  • individual_category_trajectories.png (4 subplots)")
    print(f"  • category_dominance_heatmap.png (heatmap)")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
