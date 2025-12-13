import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from scipy import stats

class OutlierDetector:
    """Multiple outlier detection methods."""
    
    @staticmethod
    def iqr_method(series: pd.Series, k: float = 1.5) -> np.ndarray:
        """IQR-based outlier detection. k=1.5 is standard; k=3 is more lenient."""
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - k * IQR
        upper = Q3 + k * IQR
        return (series < lower) | (series > upper)
    
    @staticmethod
    def zscore_method(series: pd.Series, threshold: float = 3.0) -> np.ndarray:
        """Z-score based: |z| > threshold."""
        z = np.abs((series - series.mean()) / series.std())
        return z > threshold
    
    @staticmethod
    def mad_method(series: pd.Series, threshold: float = 3.5) -> np.ndarray:
        """Median Absolute Deviation (robust to extreme outliers)."""
        median = series.median()
        mad = np.median(np.abs(series - median))
        if mad == 0:
            return np.zeros(len(series), dtype=bool)
        modified_z = 0.6745 * (series - median) / mad
        return np.abs(modified_z) > threshold
    
    @staticmethod
    def isolation_forest(df: pd.DataFrame, contamination: float = 0.1) -> np.ndarray:
        """Isolation Forest for multivariate outliers."""
        if df.shape[1] < 2:
            return np.zeros(len(df), dtype=bool)
        iso = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
        return iso.fit_predict(df) == -1
    
    @staticmethod
    def pca_outlier_detection(df: pd.DataFrame, n_components: int = 2, threshold: float = 2.0) -> np.ndarray:
        """PCA-based outlier detection using reconstruction error."""
        if df.shape[1] < 2:
            return np.zeros(len(df), dtype=bool)
        
        n_components = min(n_components, df.shape[1])
        pca = PCA(n_components=n_components, random_state=42)
        
        # Fit and transform
        X_pca = pca.fit_transform(df)
        X_reconstructed = pca.inverse_transform(X_pca)
        
        # Calculate reconstruction error
        mse = np.mean((df.values - X_reconstructed) ** 2, axis=1)
        error_threshold = np.mean(mse) + threshold * np.std(mse)
        
        return mse > error_threshold

def detect_outliers_univariate(
    df: pd.DataFrame,
    method: str = "iqr",
    iqr_k: float = 1.5,
    zscore_threshold: float = 3.0,
    mad_threshold: float = 3.5
) -> Dict[str, np.ndarray]:
    """
    Detect outliers per column using specified method.
    Returns dict: {col_name: boolean array of outlier flags}
    """
    numeric_df = df.select_dtypes(include=[np.number])
    outliers = {}
    
    for col in numeric_df.columns:
        series = numeric_df[col].dropna()
        if len(series) == 0:
            outliers[col] = np.zeros(len(df), dtype=bool)
            continue
        
        if method == "iqr":
            outliers[col] = OutlierDetector.iqr_method(series, k=iqr_k)
        elif method == "zscore":
            outliers[col] = OutlierDetector.zscore_method(series, threshold=zscore_threshold)
        elif method == "mad":
            outliers[col] = OutlierDetector.mad_method(series, threshold=mad_threshold)
        else:
            outliers[col] = np.zeros(len(series), dtype=bool)
    
    return outliers

def detect_outliers_multivariate(
    df: pd.DataFrame,
    method: str = "isolation_forest",
    contamination: float = 0.1,
    n_components: int = 2,
    pca_threshold: float = 2.0
) -> np.ndarray:
    """
    Detect multivariate outliers (considers feature correlations).
    Returns: boolean array (True = outlier)
    """
    numeric_df = df.select_dtypes(include=[np.number])
    
    if method == "isolation_forest":
        return OutlierDetector.isolation_forest(numeric_df, contamination=contamination)
    elif method == "pca":
        return OutlierDetector.pca_outlier_detection(numeric_df, n_components=n_components, threshold=pca_threshold)
    else:
        return np.zeros(len(df), dtype=bool)

def summarize_outliers(df: pd.DataFrame, outlier_flags: Dict[str, np.ndarray]) -> pd.DataFrame:
    """Summarize outlier detection across features."""
    summary = []
    for col, flags in outlier_flags.items():
        outlier_count = flags.sum()
        outlier_pct = (outlier_count / len(flags)) * 100
        summary.append({
            "feature": col,
            "outliers_count": int(outlier_count),
            "outliers_%": round(outlier_pct, 2)
        })
    return pd.DataFrame(summary).sort_values("outliers_count", ascending=False)

def identify_missing_data_patterns(df: pd.DataFrame) -> Dict:
    """Analyze missing data."""
    total_missing = df.isna().sum().sum()
    total_cells = df.shape[0] * df.shape[1]
    
    missing_by_col = df.isna().sum()
    missing_cols = missing_by_col[missing_by_col > 0].sort_values(ascending=False)
    
    return {
        "total_missing": int(total_missing),
        "total_missing_%": round((total_missing / total_cells) * 100, 2),
        "columns_with_missing": missing_cols.to_dict(),
        "rows_with_any_missing": int(df.isna().any(axis=1).sum())
    }

def get_outlier_rows(df: pd.DataFrame, outlier_flags: Dict[str, np.ndarray]) -> pd.DataFrame:
    """Get rows flagged as outliers by any feature."""
    any_outlier = np.zeros(len(df), dtype=bool)
    for flags in outlier_flags.values():
        any_outlier |= flags
    return df[any_outlier]

def calculate_qc_metrics(df: pd.DataFrame, metadata: pd.DataFrame = None) -> Dict:
    """
    Calculate quality control metrics for genomics data.
    - Library size (sum per sample)
    - Feature detection rate (% of non-zero features per sample)
    - CV (coefficient of variation) per feature
    """
    numeric_df = df.select_dtypes(include=[np.number])
    
    # Library size (sum of counts per sample)
    library_size = numeric_df.sum(axis=1)
    
    # Feature detection rate (% features > 0)
    detection_rate = (numeric_df > 0).sum(axis=1) / len(numeric_df.columns) * 100
    
    # Coefficient of variation per feature
    feature_cv = numeric_df.std() / (numeric_df.mean() + 1e-10)
    
    qc = pd.DataFrame({
        "sample": numeric_df.index,
        "library_size": library_size.values,
        "detection_rate_%": detection_rate.values,
        "n_detected_features": (numeric_df > 0).sum(axis=1).values
    })
    
    feature_qc = pd.DataFrame({
        "feature": numeric_df.columns,
        "mean_expression": numeric_df.mean().values,
        "cv": feature_cv.values,
        "detection_rate_%": (numeric_df > 0).sum(axis=0) / len(numeric_df) * 100
    })
    
    return {
        "sample_qc": qc,
        "feature_qc": feature_qc,
        "library_size_stats": {
            "min": float(library_size.min()),
            "max": float(library_size.max()),
            "median": float(library_size.median()),
            "mean": float(library_size.mean())
        }
    }