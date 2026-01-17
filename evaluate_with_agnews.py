"""
Comprehensive Model Accuracy Evaluation Script using AG News Dataset
Calculates accuracy/performance metrics for all models in TCLM pipeline
Dataset: AG News Classification Dataset from Kaggle
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from sklearn.metrics import (
    silhouette_score, 
    davies_bouldin_score,
    calinski_harabasz_score,
    mean_squared_error,
    mean_absolute_error,
    accuracy_score,
    classification_report,
    r2_score
)
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from tclm.semantic.embed import embed_corpus
from tclm.semantic.topics import extract_topics
from tclm.semantic.manifold import build_concept_trajectory
from tclm.causal.influence import estimate_causal_matrix, get_granger_summary_table
from tclm.forecast.models import forecast_trajectory
from tclm.viz.plots import plot_causal_graph


def _evaluate_forecast(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-6))) * 100
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, mape, r2


def compare_forecasters(traj: pd.DataFrame, forecast_horizon: int = 12):
    """
    Compare Auto-ARIMA vs Prophet (if available) vs naive last-value forecast.
    Saves a plot to outputs/plots/forecast_comparison.png and prints a metrics table.
    """
    if len(traj) < 12:
        print("\n[Forecast Comparison] Not enough points to compare (need 12+).")
        return
    
    # Prepare series
    y = traj["value"].astype(float).values
    dates = pd.to_datetime(traj["time_bin"])
    
    horizon = min(forecast_horizon, max(2, len(y) // 4))
    train_y = y[:-horizon]
    test_y = y[-horizon:]
    train_dates = dates[:-horizon]
    test_dates = dates[-horizon:]
    
    # Inferred frequency for future index
    try:
        freq = pd.infer_freq(dates)
    except Exception:
        freq = None
    if freq is None:
        freq = "W"
    
    forecasts = {}
    
    # Auto-ARIMA (existing)
    try:
        from pmdarima import auto_arima
        model = auto_arima(train_y, seasonal=False, error_action="ignore", suppress_warnings=True)
        fc_arima = model.predict(n_periods=horizon)
        forecasts["auto_arima"] = fc_arima
    except Exception as e:
        print(f"[Forecast Comparison] Auto-ARIMA failed: {e}")
    
    # Prophet (optional) with cmdstan backend
    try:
        from prophet import Prophet
        try:
            # Prefer explicit backend string to avoid enum issues
            backend = "CMDSTANPY"
        except Exception:
            backend = None
        df_train = pd.DataFrame({"ds": train_dates, "y": train_y})
        m = Prophet(stan_backend=backend)
        m.fit(df_train)
        future = m.make_future_dataframe(periods=horizon, freq=freq, include_history=False)
        fc_prophet = m.predict(future)["yhat"].values
        forecasts["prophet"] = fc_prophet
    except Exception as e:
        print(f"[Forecast Comparison] Prophet not available or failed: {e}")
    
    # Naive (last value)
    fc_naive = np.repeat(train_y[-1], horizon)
    forecasts["naive_last"] = fc_naive
    
    # Metrics
    rows = []
    for name, pred in forecasts.items():
        mae, rmse, mape, r2 = _evaluate_forecast(test_y, pred)
        rows.append({
            "model": name,
            "mae": mae,
            "rmse": rmse,
            "mape_%": mape,
            "r2": r2,
        })
    if rows:
        metrics_df = pd.DataFrame(rows).sort_values("mae")
        print("\n[Forecast Comparison Metrics]")
        print(metrics_df.to_string(index=False, float_format=lambda x: f"{x:0.4f}"))
    else:
        print("\n[Forecast Comparison] No forecasts computed.")
    
    # Plot
    try:
        plt.figure(figsize=(10, 5))
        plt.plot(train_dates, train_y, label="train", color="gray", alpha=0.6)
        plt.plot(test_dates, test_y, label="test", color="black", linewidth=2)
        
        start_dt = test_dates.iloc[0] if len(test_dates) else train_dates.iloc[-1]
        future_index = pd.date_range(start=start_dt, periods=horizon, freq=freq)
        colors = {"auto_arima": "tab:blue", "prophet": "tab:green", "naive_last": "tab:orange"}
        for name, pred in forecasts.items():
            plt.plot(future_index, pred, label=name, color=colors.get(name, None), linestyle="--")
        
        plt.title("Forecast Comparison")
        plt.xlabel("time")
        plt.ylabel("value")
        plt.legend()
        out_dir = Path("outputs/plots")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "forecast_comparison.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"[Forecast Comparison] Saved plot to: {out_path}")
    except Exception as e:
        import traceback
        print(f"[Forecast Comparison] Plotting failed: {repr(e)}")
        traceback.print_exc()


def load_ag_news_dataset(csv_path=None, sample_size=2000):
    """
    Load AG News dataset from CSV file
    Expected format: class,title,description
    Classes: 1=World, 2=Sports, 3=Business, 4=Sci/Tech
    """
    print("\n[Loading AG News Dataset]")
    
    if csv_path is None:
        print("  ⚠ Please provide path to train.csv or test.csv from AG News dataset")
        print("  Download from: https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset")
        
        # Create synthetic data for demonstration
        print("\n  → Creating synthetic demo dataset...")
        categories = ['World', 'Sports', 'Business', 'Sci/Tech']
        samples = []
        
        for i in range(sample_size):
            cat_idx = i % 4
            category = categories[cat_idx]
            samples.append({
                'class': cat_idx + 1,
                'category': category,
                'title': f'Sample {category} news title {i}',
                'description': f'This is a sample description for {category} news article number {i}. ' * 3,
                'text': f'Sample {category} news title {i}. This is a sample description for {category} news article number {i}. ' * 3
            })
        
        df = pd.DataFrame(samples)
        print(f"  ✓ Created {len(df)} synthetic samples")
        return df
    
    try:
        # Try reading with header first
        df = pd.read_csv(csv_path)
        
        # Check if it has headers or not
        if 'Class Index' in df.columns or 'class' in df.columns.str.lower():
            # Has headers
            df.columns = ['class', 'title', 'description']
        else:
            # No headers - reload without header
            df = pd.read_csv(csv_path, names=['class', 'title', 'description'], header=None)
        
        # Map class numbers to category names
        class_map = {1: 'World', 2: 'Sports', 3: 'Business', 4: 'Sci/Tech'}
        df['category'] = df['class'].map(class_map)
        
        # Combine title and description
        df['text'] = df['title'].fillna('').astype(str) + '. ' + df['description'].fillna('').astype(str)
        
        # Sample if too large
        if len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=42)
        
        print(f"  ✓ Loaded {len(df)} samples from {csv_path}")
        print(f"  • Categories: {df['category'].value_counts().to_dict()}")
        
        return df
        
    except Exception as e:
        print(f"  ✗ Error loading dataset: {e}")
        print(f"  → Creating synthetic demo dataset...")
        return load_ag_news_dataset(csv_path=None, sample_size=sample_size)


def add_temporal_structure(df, freq='W', months=24):
    """Add synthetic temporal structure to AG News data"""
    print("\n[Adding Temporal Structure]")
    
    start_date = datetime(2022, 1, 1)
    
    if freq == 'M':
        n_bins = months
        days_per_bin = 30
    elif freq == 'W':
        n_bins = months * 4  # Approximately 4 weeks per month
        days_per_bin = 7
    else:  # Daily
        n_bins = months * 30
        days_per_bin = 1
    
    # Assign dates evenly across bins
    dates = []
    for i in range(len(df)):
        bin_idx = i % n_bins
        date = start_date + timedelta(days=days_per_bin * bin_idx)
        dates.append(date)
    
    df['published'] = dates
    df['source'] = df['category'].apply(lambda x: 'news_' + x.lower())
    
    # Create time bins
    if freq == 'M':
        df['time_bin'] = pd.to_datetime(df['published']).dt.to_period('M').dt.to_timestamp()
    else:
        df['time_bin'] = pd.to_datetime(df['published']).dt.to_period('W').dt.to_timestamp()
    
    print(f"  • Frequency: {freq}")
    print(f"  • Time bins created: {df['time_bin'].nunique()}")
    print(f"  • Date range: {df['time_bin'].min()} to {df['time_bin'].max()}")
    
    return df


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
    
    # 2. Cosine similarity distribution
    from sklearn.metrics.pairwise import cosine_similarity
    cos_sim_matrix = cosine_similarity(X[:100])  # Sample for speed
    np.fill_diagonal(cos_sim_matrix, np.nan)
    avg_sim = np.nanmean(cos_sim_matrix)
    std_sim = np.nanstd(cos_sim_matrix)
    print(f"  • Average pairwise cosine similarity: {avg_sim:.4f} ± {std_sim:.4f}")
    
    # 3. Category discrimination (using ground truth labels)
    print("\n[Category Discrimination - Ground Truth Evaluation]")
    if 'category' in df_embedded.columns:
        labels = df_embedded['category'].astype('category').cat.codes.values
        
        sil = silhouette_score(X, labels, metric='cosine')
        db = davies_bouldin_score(X, labels)
        ch = calinski_harabasz_score(X, labels)
        
        print(f"  • Silhouette Score: {sil:.4f} (range [-1, 1], higher better)")
        print(f"  • Davies-Bouldin Index: {db:.4f} (lower better)")
        print(f"  • Calinski-Harabasz Index: {ch:.2f} (higher better)")
        
        # Classification accuracy using k-NN
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.model_selection import cross_val_score
        
        knn = KNeighborsClassifier(n_neighbors=5, metric='cosine')
        cv_scores = cross_val_score(knn, X, labels, cv=min(5, len(set(labels))))
        
        print(f"\n[k-NN Classification Accuracy]")
        print(f"  • 5-Fold CV Accuracy: {cv_scores.mean():.2%} ± {cv_scores.std():.2%}")
        print(f"  • This measures how well embeddings preserve category information")
        
        embedding_accuracy = cv_scores.mean()
    else:
        embedding_accuracy = 0.5
    
    print(f"\n  ✓ EMBEDDING MODEL ACCURACY: {embedding_accuracy:.2%}")
    
    return embedding_accuracy


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
    predicted_labels = kmeans.fit_predict(X)
    
    # Internal validation metrics
    print("\n[Internal Validation Metrics]")
    sil = silhouette_score(X, predicted_labels)
    db = davies_bouldin_score(X, predicted_labels)
    
    print(f"  • Silhouette Score: {sil:.4f}")
    print(f"  • Davies-Bouldin Index: {db:.4f}")
    
    # External validation (if ground truth available)
    if 'category' in df_embedded.columns:
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
        
        true_labels = df_embedded['category'].astype('category').cat.codes.values
        
        ari = adjusted_rand_score(true_labels, predicted_labels)
        nmi = normalized_mutual_info_score(true_labels, predicted_labels)
        
        print(f"\n[External Validation (vs Ground Truth)]")
        print(f"  • Adjusted Rand Index: {ari:.4f} (range [-1, 1])")
        print(f"  • Normalized Mutual Info: {nmi:.4f} (range [0, 1])")
        
        # Purity score
        from scipy.optimize import linear_sum_assignment
        confusion_matrix = np.zeros((n_clusters, len(set(true_labels))))
        for cluster_id in range(n_clusters):
            cluster_mask = predicted_labels == cluster_id
            for true_label in set(true_labels):
                confusion_matrix[cluster_id, true_label] = np.sum(
                    (predicted_labels == cluster_id) & (true_labels == true_label)
                )
        
        row_ind, col_ind = linear_sum_assignment(-confusion_matrix)
        purity = confusion_matrix[row_ind, col_ind].sum() / len(predicted_labels)
        
        print(f"  • Cluster Purity: {purity:.4f}")
        
        clustering_accuracy = (nmi + purity) / 2
    else:
        clustering_accuracy = (sil + 1) / 2
    
    print(f"\n  ✓ CLUSTERING ACCURACY: {clustering_accuracy:.2%}")
    
    return clustering_accuracy


def evaluate_causal_model(df_embedded, causal_results):
    """Evaluate VAR/Granger causal inference model"""
    print("\n" + "="*70)
    print("3. VAR CAUSAL INFERENCE MODEL EVALUATION")
    print("="*70)
    
    n_bins = df_embedded['time_bin'].nunique()
    n_sources = df_embedded['source'].nunique()
    
    p_mat = getattr(causal_results, "p_value_matrix", None)
    f_mat = getattr(causal_results, "f_statistic_matrix", None)
    
    shape_str = p_mat.shape if p_mat is not None else ("N/A",)
    
    print(f"\n[Data Configuration]")
    print(f"  • Number of time bins: {n_bins}")
    print(f"  • Number of sources: {n_sources}")
    print(f"  • Causal matrix shape: {shape_str}")
    
    if p_mat is not None and p_mat.size > 1:
        # Matrix properties
        print(f"\n[Causal Matrix Properties]")
        print(f"  • Min p-value: {np.nanmin(p_mat):.4f}")
        print(f"  • Mean p-value: {np.nanmean(p_mat):.4f}")
        print(f"  • Max p-value: {np.nanmax(p_mat):.4f}")
        if f_mat is not None:
            print(f"  • Max F-statistic: {np.nanmax(f_mat):.4f}")
    else:
        print("\n[Causal Matrix Properties]")
        print("  • Not enough data to compute causal matrix")
    
    # Confidence based on sample size
    confidence = min(0.9, max(0.1, n_bins / 50))
    
    print(f"\n  ✓ MODEL CONFIDENCE: {confidence:.2%}")
    print(f"    → Based on {n_bins} time points")
    
    return confidence


def evaluate_forecasting_model(traj):
    """Evaluate Auto-ARIMA forecasting model with walk-forward validation"""
    print("\n" + "="*70)
    print("4. AUTO-ARIMA FORECASTING MODEL EVALUATION")
    print("="*70)
    
    n_points = len(traj)
    print(f"\n[Data Configuration]")
    print(f"  • Number of observations: {n_points}")
    
    if n_points < 10:
        print(f"\n  ⚠ Only {n_points} observations - need 10+ for reliable evaluation")
        return None
    
    # Walk-forward validation
    print("\n[Walk-Forward Cross-Validation]")
    y = traj["value"].astype(float).values
    
    min_train_size = max(5, int(n_points * 0.6))
    forecasts = []
    actuals = []
    
    for i in range(min_train_size, n_points):
        train = traj.iloc[:i]
        actual = y[i]
        pred = forecast_trajectory(train, steps=1)[0]
        forecasts.append(pred)
        actuals.append(actual)
    
    forecasts = np.array(forecasts)
    actuals = np.array(actuals)
    
    # Metrics
    mae = mean_absolute_error(actuals, forecasts)
    rmse = np.sqrt(mean_squared_error(actuals, forecasts))
    mape = np.mean(np.abs((actuals - forecasts) / (np.abs(actuals) + 1e-6))) * 100
    
    # R-squared
    ss_res = np.sum((actuals - forecasts) ** 2)
    ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
    r2 = max(0, 1 - (ss_res / (ss_tot + 1e-10)))
    
    # Directional accuracy
    if len(actuals) > 1:
        actual_dir = np.diff(actuals) > 0
        forecast_dir = np.diff(forecasts) > 0
        dir_acc = np.mean(actual_dir == forecast_dir)
    else:
        dir_acc = 0.5
    
    print(f"  • Mean Absolute Error (MAE): {mae:.4f}")
    print(f"  • Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"  • MAPE: {mape:.2f}%")
    print(f"  • R² Score: {r2:.4f}")
    print(f"  • Directional Accuracy: {dir_acc:.2%}")
    
    forecasting_accuracy = r2
    print(f"\n  ✓ FORECASTING ACCURACY (R²): {forecasting_accuracy:.2%}")
    
    return forecasting_accuracy


def main():
    print("\n" + "="*70)
    print("TCLM MODEL ACCURACY EVALUATION - AG NEWS DATASET")
    print("="*70)
    
    # Try to load AG News dataset
    print("\n" + "="*70)
    print("DATASET LOADING")
    print("="*70)
    print("\nTo use real AG News data:")
    print("  1. Download from: https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset")
    print("  2. Extract train.csv")
    print("  3. Run: python evaluate_with_agnews.py <path_to_train.csv>")
    
    import sys
    csv_path = sys.argv[1] if len(sys.argv) > 1 else None
    
    # Load dataset (use more samples for better statistics)
    df = load_ag_news_dataset(csv_path=csv_path, sample_size=5000)
    
    # Add temporal structure with more time bins for better forecasting
    df = add_temporal_structure(df, freq='W', months=24)  # Weekly bins over 24 months = ~104 time points
    
    # Generate embeddings
    print("\n[Generating Embeddings...]")
    embedded = embed_corpus(df)
    print(f"  ✓ Generated {len(embedded)} embeddings")
    
    # Extract topics
    print("\n[Extracting Topics...]")
    topics = extract_topics(embedded, n_topics=8)
    print(f"  ✓ Extracted {len(topics)} topics")
    
    # Build trajectory
    concept = "technology"
    print(f"\n[Building Trajectory for '{concept}'...]")
    traj = build_concept_trajectory(embedded, concept=concept)
    print(f"  ✓ Built trajectory with {len(traj)} time points")
    
    # Causal matrix
    print("\n[Estimating Causal Influences...]")
    causal = estimate_causal_matrix(embedded)
    p_shape = getattr(causal, "p_value_matrix", np.array([[0]])).shape
    print(f"  ✓ Generated causal matrix {p_shape}")
    
    # Show full Granger table (all p-values) for inspection
    full_table = get_granger_summary_table(causal, significance_level=1.0)
    if not full_table.empty:
        print("\n[Full Granger Table (no filtering)]")
        print(full_table)
    else:
        print("\n[Full Granger Table] No causal pairs computed.")
    
    # Save causal graph visualization with a looser threshold to surface weaker links
    significance_level = 0.40  # adjust here to relax/tighten displayed edges
    graph_path = plot_causal_graph(causal, significance_level=significance_level)
    print(f"  ✓ Saved causal graph to: {graph_path} (p < {significance_level})")
    
    # Run all evaluations
    print("\n" + "="*70)
    print("RUNNING MODEL EVALUATIONS")
    print("="*70)
    
    results = {}
    
    results['embedding'] = evaluate_embedding_model(embedded)
    results['clustering'] = evaluate_clustering_model(embedded, topics)
    results['causal'] = evaluate_causal_model(embedded, causal)
    results['forecasting'] = evaluate_forecasting_model(traj)
    
    # Compare forecasting models (Auto-ARIMA vs Prophet (if available) vs naive)
    compare_forecasters(traj, forecast_horizon=12)
    
    # Summary
    print("\n" + "="*70)
    print("FINAL ACCURACY SUMMARY")
    print("="*70)
    
    for model, score in results.items():
        if score is not None:
            bar = '█' * int(score * 50)
            print(f"  {model.upper():20s}: {score:6.2%} [{bar}]")
        else:
            print(f"  {model.upper():20s}: N/A (insufficient data)")
    
    # Overall system score
    valid_scores = [s for s in results.values() if s is not None]
    if valid_scores:
        overall = np.mean(valid_scores)
        bar = '█' * int(overall * 50)
        print(f"\n  {'OVERALL SYSTEM':20s}: {overall:6.2%} [{bar}]")
        
        if overall >= 0.8:
            grade = "Excellent ⭐⭐⭐⭐⭐"
        elif overall >= 0.6:
            grade = "Good ⭐⭐⭐⭐"
        elif overall >= 0.4:
            grade = "Fair ⭐⭐⭐"
        else:
            grade = "Needs Improvement ⭐⭐"
        
        print(f"  {'GRADE':20s}: {grade}")
    
    print("\n" + "="*70)
    print("NOTES:")
    print("="*70)
    print("  • Embedding: k-NN classification accuracy on category labels")
    print("  • Clustering: NMI + Purity score vs ground truth categories")
    print("  • Causal: Confidence based on time series length")
    print("  • Forecasting: R² score from walk-forward cross-validation")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
