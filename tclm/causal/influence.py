from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import grangercausalitytests
from typing import Tuple, Dict, List, NamedTuple


class GrangerResult(NamedTuple):
    """Results from Granger causality test"""
    p_value: float
    f_statistic: float
    granger_causes: bool
    all_p_values: List[float]


class CausalResults(NamedTuple):
    """Complete causal analysis results"""
    p_value_matrix: np.ndarray
    f_statistic_matrix: np.ndarray
    sources: List[str]
    granger_results: Dict[Tuple[str, str], GrangerResult]
    fitted_var: VAR


def _source_series(df: pd.DataFrame, sources: list[str]) -> pd.DataFrame:
    rows = []
    for time_bin, g in df.groupby("time_bin"):
        row = {"time_bin": time_bin}
        for s in sources:
            sub = g[g["source"] == s]
            if len(sub) == 0:
                row[s] = np.nan
            else:
                row[s] = float(np.stack(sub["embedding"].values).mean())
        rows.append(row)
    X = pd.DataFrame(rows).sort_values("time_bin").reset_index(drop=True)
    X = X.ffill().bfill()
    return X.set_index("time_bin")


def _test_granger_causality(fitted_var, causing_var: str, caused_var: str, max_lag: int) -> GrangerResult:
    """
    Test if one variable Granger-causes another.
    
    Args:
        fitted_var: Fitted VAR model
        causing_var: Name of the variable that might cause
        caused_var: Name of the variable that might be caused
        max_lag: Maximum lag to test
        
    Returns:
        GrangerResult with p-value, F-statistic, and decision
    """
    try:
        # Use the VAR model's test_causality method if available
        # This tests if causing_var Granger-causes caused_var
        var_names = fitted_var.names
        if causing_var not in var_names or caused_var not in var_names:
            return GrangerResult(p_value=1.0, f_statistic=0.0, granger_causes=False, all_p_values=[])
        
        # Get indices
        causing_idx = var_names.index(causing_var)
        caused_idx = var_names.index(caused_var)
        
        # Check if we have enough data points
        n_obs = len(fitted_var.endog)
        min_obs_required = max_lag * 3  # Need at least 3x the lag order
        if n_obs < min_obs_required:
            # Insufficient data for reliable test
            return GrangerResult(p_value=1.0, f_statistic=0.0, granger_causes=False, all_p_values=[])
        
        # Test causality using the fitted model's method
        # We test if causing_var (columns) Granger-causes caused_var (rows)
        # The test_causality method tests: does causing_var help predict caused_var?
        try:
            test_result = fitted_var.test_causality(caused_idx, causing_idx, kind='f')
            p_value = test_result.pvalue
            f_statistic = test_result.test_statistic
            
            # Validate results
            if np.isnan(p_value) or np.isinf(p_value) or np.isnan(f_statistic) or np.isinf(f_statistic):
                # Invalid test results, try fallback
                raise ValueError("Invalid test results")
            
            # Check for reasonable F-statistic (should be positive)
            if f_statistic < 0:
                # Negative F-statistic indicates test failure
                raise ValueError("Negative F-statistic")
            
            granger_causes = p_value < 0.10
            
            return GrangerResult(
                p_value=float(p_value),
                f_statistic=float(f_statistic),
                granger_causes=granger_causes,
                all_p_values=[float(p_value)]  # Single p-value from the test
            )
        except (AttributeError, ValueError, Exception) as e:
            # Fallback: use grangercausalitytests directly
            try:
                # Get the data for these two variables
                data = fitted_var.endog[[causing_var, caused_var]]
                
                # Ensure we have enough data
                if len(data) < max_lag * 2:
                    return GrangerResult(p_value=1.0, f_statistic=0.0, granger_causes=False, all_p_values=[])
                
                # Run Granger causality test
                # grangercausalitytests tests if the second variable causes the first
                # So we pass [caused_var, causing_var] to test if causing_var causes caused_var
                results = grangercausalitytests(data[[caused_var, causing_var]], max_lag, verbose=False)
                
                # Extract p-values for each lag (using ssr_ftest)
                p_values = []
                f_stats = []
                for lag in range(1, max_lag + 1):
                    if lag in results:
                        test_result = results[lag][0]
                        if 'ssr_ftest' in test_result:
                            f_stat = test_result['ssr_ftest'][0]
                            p_val = test_result['ssr_ftest'][1]
                            # Validate values
                            if not (np.isnan(f_stat) or np.isinf(f_stat) or np.isnan(p_val) or np.isinf(p_val)):
                                if f_stat >= 0:  # F-statistic should be non-negative
                                    p_values.append(p_val)
                                    f_stats.append(f_stat)
                
                if not p_values:
                    return GrangerResult(p_value=1.0, f_statistic=0.0, granger_causes=False, all_p_values=[])
                
                # Use minimum p-value across lags
                min_p_idx = np.argmin(p_values)
                min_p_value = p_values[min_p_idx]
                f_statistic = f_stats[min_p_idx]
                
                granger_causes = min_p_value < 0.10
                
                return GrangerResult(
                    p_value=min_p_value,
                    f_statistic=f_statistic,
                    granger_causes=granger_causes,
                    all_p_values=p_values
                )
            except Exception:
                # If fallback also fails, return no causality
                return GrangerResult(p_value=1.0, f_statistic=0.0, granger_causes=False, all_p_values=[])
    except Exception:
        # If test fails, return no causality
        return GrangerResult(p_value=1.0, f_statistic=0.0, granger_causes=False, all_p_values=[])


def estimate_causal_matrix(df: pd.DataFrame, significance_level: float = 0.10) -> CausalResults:
    """
    Estimate causal matrix using VAR + Granger causality tests.
    
    Args:
        df: DataFrame with 'source', 'time_bin', and 'embedding' columns
        significance_level: P-value threshold for significance (default 0.10 for 90% confidence)
        
    Returns:
        CausalResults with p-value matrix, F-statistics, and detailed results
    """
    sources = list(df["source"].unique())
    n = len(sources)
    
    if n < 2:
        # Return empty results
        empty_mat = np.ones((1, 1))
        return CausalResults(
            p_value_matrix=empty_mat,
            f_statistic_matrix=empty_mat,
            sources=sources if sources else ["unknown"],
            granger_results={},
            fitted_var=None
        )
    
    X = _source_series(df, sources)
    
    # Check for constant columns and add small noise if needed
    for col in X.columns:
        if X[col].std() < 1e-10:
            X[col] = X[col] + np.random.normal(0, 1e-6, len(X))
    
    # Check if we have enough time points for VAR analysis
    # VAR models need at least 10-15 observations, preferably more
    min_time_points = 10
    if len(X) < min_time_points:
        # Insufficient data for VAR analysis
        p_mat = np.ones((n, n))
        f_mat = np.zeros((n, n))
        np.fill_diagonal(p_mat, 0.0)
        return CausalResults(
            p_value_matrix=p_mat,
            f_statistic_matrix=f_mat,
            sources=sources,
            granger_results={},
            fitted_var=None
        )
    
    # Fit VAR model
    model = VAR(X)
    try:
        # Adjust maxlags based on available data
        # Need at least 3*maxlags observations for reliable estimation
        maxlags = min(4, max(1, (len(X) - 1) // 3))
        if maxlags < 1:
            maxlags = 1
        
        fitted = model.fit(maxlags=maxlags, trend='n')
    except Exception as e:
        # If VAR fails, return matrices with no causality
        p_mat = np.ones((n, n))
        f_mat = np.zeros((n, n))
        np.fill_diagonal(p_mat, 0.0)
        return CausalResults(
            p_value_matrix=p_mat,
            f_statistic_matrix=f_mat,
            sources=sources,
            granger_results={},
            fitted_var=None
        )
    
    # Initialize result matrices
    p_value_matrix = np.ones((n, n))
    f_statistic_matrix = np.zeros((n, n))
    granger_results = {}
    
    # Test all pairs for Granger causality
    for i, source_i in enumerate(sources):
        for j, source_j in enumerate(sources):
            if i == j:
                p_value_matrix[i, j] = 0.0
                continue
            
            # Test if source_i Granger-causes source_j
            result = _test_granger_causality(fitted, source_i, source_j, maxlags)
            
            p_value_matrix[i, j] = result.p_value
            f_statistic_matrix[i, j] = result.f_statistic
            granger_results[(source_i, source_j)] = result
    
    return CausalResults(
        p_value_matrix=p_value_matrix,
        f_statistic_matrix=f_statistic_matrix,
        sources=sources,
        granger_results=granger_results,
        fitted_var=fitted
    )


def get_granger_summary_table(causal_results: CausalResults, significance_level: float = 0.10) -> pd.DataFrame:
    """
    Generate a summary table of Granger causality test results.
    
    Args:
        causal_results: Results from estimate_causal_matrix
        significance_level: P-value threshold for significance
        
    Returns:
        DataFrame with columns: Cause, Effect, P-Value, F-Statistic, Significant, Confidence
    """
    rows = []
    sources = causal_results.sources
    p_matrix = causal_results.p_value_matrix
    f_matrix = causal_results.f_statistic_matrix
    
    for i, cause in enumerate(sources):
        for j, effect in enumerate(sources):
            if i == j:
                continue
            
            p_val = p_matrix[i, j]
            f_stat = f_matrix[i, j]
            
            # Handle invalid values
            if np.isnan(p_val) or np.isinf(p_val):
                p_val = 1.0  # No causality if test failed
            if np.isnan(f_stat) or np.isinf(f_stat) or f_stat < 0:
                f_stat = 0.0  # Invalid F-statistic
            
            # Clamp p-value to valid range
            p_val = max(0.0, min(1.0, p_val))
            
            significant = p_val < significance_level
            confidence = (1 - p_val) * 100
            
            rows.append({
                'Cause': cause,
                'Effect': effect,
                'P-Value': p_val,
                'F-Statistic': f_stat,
                'Significant': 'Yes' if significant else 'No',
                'Confidence (%)': confidence
            })
    
    df = pd.DataFrame(rows)
    # Sort by p-value (most significant first)
    df = df.sort_values('P-Value')
    return df


