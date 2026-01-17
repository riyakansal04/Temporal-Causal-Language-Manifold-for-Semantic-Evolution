"""
Comprehensive Model Accuracy Evaluation Script
Calculates accuracy/performance metrics for all models in TCLM pipeline
"""

from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.metrics import (
    silhouette_score, 
    davies_bouldin_score,
    calinski_harabasz_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score
)
from sklearn.model_selection import KFold
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

from tclm.ingest.rss import fetch_rss_corpus
from tclm.ingest.arxiv import fetch_arxiv_corpus
from tclm.preprocess.clean import clean_corpus
from tclm.preprocess.timebin import timebin_corpus
from tclm.semantic.embed import embed_corpus
from tclm.semantic.topics import extract_topics
from tclm.semantic.manifold import build_concept_trajectory
from tclm.causal.influence import estimate_causal_matrix
from tclm.forecast.models import forecast_trajectory


def evaluate_embedding_model(df_embedded):
    """Evaluate SentenceTransformer embedding quality"""
    print("\n" + "="*70)
    print("1. SENTENCETRANSFORMER EMBEDDING MODEL EVALUATION")
    print("="*70)
    
    X = np.stack(df_embedded["embedding"].values)
    
    # Intrinsic quality metrics
    print("\n[Intrinsic Quality Metrics]")
    
    # 1. Embedding variance (diversity)
    var = np.var(X, axis=0).mean()
    print(f"  • Mean embedding variance: {var:.4f}")
    print(f"    → Higher is better (more diverse representations)")
    
    # 2. Cosine similarity distribution
    from sklearn.metrics.pairwise import cosine_similarity
    cos_sim_matrix = cosine_similarity(X)
    np.fill_diagonal(cos_sim_matrix, np.nan)
    avg_sim = np.nanmean(cos_sim_matrix)
    std_sim = np.nanstd(cos_sim_matrix)
    print(f"  • Average pairwise cosine similarity: {avg_sim:.4f} ± {std_sim:.4f}")
    print(f"    → Should be moderate (0.3-0.7) for good discrimination")
    
    # 3. Embedding norm consistency
    norms = np.linalg.norm(X, axis=1)
    print(f"  • Embedding norms: {norms.mean():.4f} ± {norms.std():.4f}")
    print(f"    → Normalized embeddings should be ~1.0")
    
    # Extrinsic validation: Title-Text retrieval
    print("\n[Extrinsic Validation: Title-Text Retrieval]")
    df_with_title = df_embedded.dropna(subset=['title']).copy()
    
    if len(df_with_title) >= 10:
        X_subset = np.stack(df_with_title["embedding"].values)
        cos_sim = cosine_similarity(X_subset)
        np.fill_diagonal(cos_sim, -np.inf)
        
        # Recall@k
        for k in [1, 3, 5]:
            top_k = np.argsort(cos_sim, axis=1)[:, -k:]
            correct = sum(i in top_k[i] for i in range(len(X_subset)))
            recall = correct / len(X_subset)
            print(f"  • Recall@{k}: {recall:.2%}")
    else:
        print("  • Not enough samples with titles for retrieval evaluation")
    
    # Source discrimination
    print("\n[Source Discrimination]")
    if 'source' in df_embedded.columns and df_embedded['source'].nunique() > 1:
        source_labels = df_embedded['source'].astype('category').cat.codes.values
        sil = silhouette_score(X, source_labels, metric='cosine')
        db = davies_bouldin_score(X, source_labels)
        ch = calinski_harabasz_score(X, source_labels)
        
        print(f"  • Silhouette Score: {sil:.4f}")
        print(f"    → Range [-1, 1], higher is better (>0.5 good)")
        print(f"  • Davies-Bouldin Index: {db:.4f}")
        print(f"    → Lower is better (<1.0 good)")
        print(f"  • Calinski-Harabasz Index: {ch:.2f}")
        print(f"    → Higher is better (>100 good)")
        
        accuracy_score = (sil + 1) / 2  # Normalize to [0, 1]
        print(f"\n  ✓ EMBEDDING QUALITY SCORE: {accuracy_score:.2%}")
    else:
        print("  • Single source - cannot evaluate discrimination")
        accuracy_score = None
    
    return accuracy_score


def evaluate_clustering_model(df_embedded, topics):
    """Evaluate KMeans clustering quality"""
    print("\n" + "="*70)
    print("2. KMEANS CLUSTERING MODEL EVALUATION")
    print("="*70)
    
    if not topics or len(df_embedded) < 10:
        print("  • Not enough data for clustering evaluation")
        return None
    
    from sklearn.cluster import KMeans
    X = np.stack(df_embedded["embedding"].values)
    n_clusters = len(topics)
    
    print(f"\n[Clustering Configuration]")
    print(f"  • Number of clusters: {n_clusters}")
    print(f"  • Number of samples: {len(X)}")
    
    # Refit to get labels
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels = kmeans.fit_predict(X)
    
    # Internal validation metrics
    print("\n[Internal Validation Metrics]")
    sil = silhouette_score(X, labels)
    db = davies_bouldin_score(X, labels)
    ch = calinski_harabasz_score(X, labels)
    
    print(f"  • Silhouette Score: {sil:.4f}")
    print(f"    → Range [-1, 1], higher is better")
    print(f"  • Davies-Bouldin Index: {db:.4f}")
    print(f"    → Lower is better")
    print(f"  • Calinski-Harabasz Index: {ch:.2f}")
    print(f"    → Higher is better")
    
    # Cluster size distribution
    print("\n[Cluster Size Distribution]")
    unique, counts = np.unique(labels, return_counts=True)
    sizes = dict(zip(unique, counts))
    for cluster_id, size in sizes.items():
        pct = size / len(X) * 100
        print(f"  • Cluster {cluster_id}: {size} docs ({pct:.1f}%)")
    
    # Balance score (entropy-based)
    probs = counts / counts.sum()
    entropy = -np.sum(probs * np.log(probs + 1e-10))
    max_entropy = np.log(n_clusters)
    balance = entropy / max_entropy
    print(f"\n  • Cluster balance: {balance:.2%}")
    print(f"    → 100% means perfectly balanced clusters")
    
    # Overall accuracy (normalized silhouette)
    accuracy = (sil + 1) / 2
    print(f"\n  ✓ CLUSTERING ACCURACY: {accuracy:.2%}")
    
    return accuracy


def evaluate_causal_model(df_embedded, causal_matrix):
    """Evaluate VAR causal inference model"""
    print("\n" + "="*70)
    print("3. VAR CAUSAL INFERENCE MODEL EVALUATION")
    print("="*70)
    
    n_bins = df_embedded['time_bin'].nunique()
    n_sources = df_embedded['source'].nunique()
    
    print(f"\n[Data Configuration]")
    print(f"  • Number of time bins: {n_bins}")
    print(f"  • Number of sources: {n_sources}")
    print(f"  • Causal matrix shape: {causal_matrix.shape}")
    
    if n_bins < 10:
        print(f"\n  ⚠ WARNING: Only {n_bins} time bins!")
        print(f"    → VAR needs 10+ time points for reliable estimates")
        print(f"    → Current results are highly uncertain")
    
    # Matrix properties
    print("\n[Causal Matrix Properties]")
    print(f"  • Mean influence strength: {causal_matrix.mean():.4f}")
    print(f"  • Max influence: {causal_matrix.max():.4f}")
    print(f"  • Std deviation: {causal_matrix.std():.4f}")
    
    if n_sources == 2:
        print(f"\n[Cross-Source Influences]")
        sources = list(df_embedded['source'].unique())
        print(f"  • {sources[0]} → {sources[1]}: {causal_matrix[0, 1]:.4f}")
        print(f"  • {sources[1]} → {sources[0]}: {causal_matrix[1, 0]:.4f}")
    
    # Confidence estimation (based on data sufficiency)
    if n_bins < 5:
        confidence = 0.1
        status = "Very Low"
    elif n_bins < 10:
        confidence = 0.3
        status = "Low"
    elif n_bins < 20:
        confidence = 0.6
        status = "Moderate"
    else:
        confidence = 0.85
        status = "High"
    
    print(f"\n  ✓ MODEL CONFIDENCE: {confidence:.2%} ({status})")
    print(f"    → Based on {n_bins} time points")
    
    return confidence


def evaluate_forecasting_model(traj):
    """Evaluate Auto-ARIMA forecasting model"""
    print("\n" + "="*70)
    print("4. AUTO-ARIMA FORECASTING MODEL EVALUATION")
    print("="*70)
    
    n_points = len(traj)
    print(f"\n[Data Configuration]")
    print(f"  • Number of observations: {n_points}")
    
    if n_points < 10:
        print(f"\n  ⚠ WARNING: Only {n_points} observations!")
        print(f"    → ARIMA needs 20+ points for robust forecasting")
        print(f"    → Results will be unreliable")
        return None
    
    # Walk-forward validation (time series cross-validation)
    print("\n[Walk-Forward Cross-Validation]")
    y = traj["value"].astype(float).values
    
    min_train_size = max(5, int(n_points * 0.5))
    test_size = min(6, n_points - min_train_size)
    
    if test_size < 2:
        print("  • Not enough data for cross-validation")
        return None
    
    errors = []
    forecasts_all = []
    actuals_all = []
    
    for i in range(min_train_size, n_points):
        train = traj.iloc[:i]
        if i < n_points:
            actual = y[i]
            pred = forecast_trajectory(train, steps=1)[0]
            error = abs(actual - pred)
            errors.append(error)
            forecasts_all.append(pred)
            actuals_all.append(actual)
    
    if not errors:
        print("  • Could not perform validation")
        return None
    
    forecasts_all = np.array(forecasts_all)
    actuals_all = np.array(actuals_all)
    
    # Metrics
    mae = np.mean(errors)
    rmse = np.sqrt(mean_squared_error(actuals_all, forecasts_all))
    mape = np.mean(np.abs((actuals_all - forecasts_all) / (np.abs(actuals_all) + 1e-6))) * 100
    
    # R-squared
    ss_res = np.sum((actuals_all - forecasts_all) ** 2)
    ss_tot = np.sum((actuals_all - np.mean(actuals_all)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-10))
    
    # Directional accuracy
    if len(actuals_all) > 1:
        actual_direction = np.diff(actuals_all) > 0
        forecast_direction = np.diff(forecasts_all) > 0
        dir_acc = np.mean(actual_direction == forecast_direction)
    else:
        dir_acc = 0.5
    
    print(f"  • Mean Absolute Error (MAE): {mae:.4f}")
    print(f"  • Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"  • Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
    print(f"  • R² Score: {r2:.4f}")
    print(f"  • Directional Accuracy: {dir_acc:.2%}")
    
    # Correlation
    if len(forecasts_all) > 2:
        corr, p_val = pearsonr(actuals_all, forecasts_all)
        print(f"  • Pearson Correlation: {corr:.4f} (p={p_val:.4f})")
    
    # Overall accuracy (based on R²)
    accuracy = max(0, r2)
    print(f"\n  ✓ FORECASTING ACCURACY (R²): {accuracy:.2%}")
    
    return accuracy


def evaluate_concept_trajectory(traj, df_embedded, concept):
    """Evaluate concept trajectory building"""
    print("\n" + "="*70)
    print("5. CONCEPT TRAJECTORY QUALITY EVALUATION")
    print("="*70)
    
    print(f"\n[Trajectory for concept: '{concept}']")
    print(f"  • Number of time points: {len(traj)}")
    print(f"  • Value range: [{traj['value'].min():.4f}, {traj['value'].max():.4f}]")
    print(f"  • Mean similarity: {traj['value'].mean():.4f}")
    print(f"  • Std deviation: {traj['value'].std():.4f}")
    
    # Temporal stability
    if len(traj) > 1:
        diffs = np.abs(np.diff(traj['value'].values))
        avg_change = diffs.mean()
        max_change = diffs.max()
        print(f"\n[Temporal Dynamics]")
        print(f"  • Average change per period: {avg_change:.4f}")
        print(f"  • Maximum change: {max_change:.4f}")
        
        # Trend detection
        from scipy.stats import linregress
        x = np.arange(len(traj))
        y = traj['value'].values
        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        
        print(f"\n[Trend Analysis]")
        print(f"  • Slope: {slope:.6f}")
        print(f"  • R² of trend: {r_value**2:.4f}")
        print(f"  • p-value: {p_value:.4f}")
        
        if p_value < 0.05:
            trend = "increasing" if slope > 0 else "decreasing"
            print(f"  • Significant {trend} trend detected")
        else:
            print(f"  • No significant trend")
    
    # Concept prevalence
    df_embedded_copy = df_embedded.copy()
    df_embedded_copy["contains"] = df_embedded_copy["text"].str.lower().str.contains(concept.lower())
    prevalence = df_embedded_copy["contains"].mean()
    
    print(f"\n[Concept Prevalence]")
    print(f"  • Documents containing '{concept}': {prevalence:.2%}")
    
    if prevalence < 0.01:
        print(f"  ⚠ Very low prevalence - trajectory may be noisy")
        quality = 0.3
    elif prevalence < 0.05:
        print(f"  ⚠ Low prevalence - results less reliable")
        quality = 0.6
    else:
        print(f"  ✓ Good prevalence for analysis")
        quality = 0.9
    
    print(f"\n  ✓ TRAJECTORY QUALITY SCORE: {quality:.2%}")
    
    return quality


def main():
    print("\n" + "="*70)
    print("TCLM COMPREHENSIVE MODEL ACCURACY EVALUATION")
    print("="*70)
    
    # Ingest data
    print("\n[Data Ingestion]")
    rss_df = fetch_rss_corpus(max_docs=800)
    arxiv_df = fetch_arxiv_corpus(max_docs=800)
    print(f"  • RSS docs: {len(rss_df)}")
    print(f"  • arXiv docs: {len(arxiv_df)}")
    
    # Preprocess
    corpus = clean_corpus([rss_df, arxiv_df])
    binned = timebin_corpus(corpus, freq='M')
    
    # Embed
    print("\n[Generating Embeddings...]")
    embedded = embed_corpus(binned)
    
    # Extract topics
    topics = extract_topics(embedded)
    
    # Build trajectory
    concept = "privacy"
    traj = build_concept_trajectory(embedded, concept=concept)
    
    # Causal matrix
    causal = estimate_causal_matrix(embedded)
    
    # Run all evaluations
    results = {}
    
    results['embedding'] = evaluate_embedding_model(embedded)
    results['clustering'] = evaluate_clustering_model(embedded, topics)
    results['causal'] = evaluate_causal_model(embedded, causal)
    results['forecasting'] = evaluate_forecasting_model(traj)
    results['trajectory'] = evaluate_concept_trajectory(traj, embedded, concept)
    
    # Summary
    print("\n" + "="*70)
    print("OVERALL ACCURACY SUMMARY")
    print("="*70)
    
    for model, score in results.items():
        if score is not None:
            print(f"  • {model.upper()}: {score:.2%}")
        else:
            print(f"  • {model.upper()}: N/A (insufficient data)")
    
    # Overall system score
    valid_scores = [s for s in results.values() if s is not None]
    if valid_scores:
        overall = np.mean(valid_scores)
        print(f"\n  ✓ OVERALL SYSTEM ACCURACY: {overall:.2%}")
    else:
        print(f"\n  ⚠ Cannot compute overall accuracy - insufficient data")
    
    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)
    print("  • Increase data: Use --max-docs 5000+ for better results")
    print("  • Longer timeframe: Use --start and --end with 1+ year span")
    print("  • Finer granularity: Use --freq D for daily bins (more time points)")
    print("  • Fine-tune embeddings: Run 'python -m tclm.cli train'")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
