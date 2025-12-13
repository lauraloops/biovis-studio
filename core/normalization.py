import pandas as pd
import numpy as np
from typing import Tuple, Dict
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, PowerTransformer
from scipy import stats

def normalize_data(
    df: pd.DataFrame,
    method: str = "standard",
    per_feature: bool = True
) -> Tuple[pd.DataFrame, dict]:
    """
    Normalize numeric columns using specified method.
    
    Methods:
    - "standard": (x - mean) / std
    - "robust": (x - median) / IQR (resists outliers)
    - "minmax": (x - min) / (max - min) → [0,1]
    - "log1p": log(1 + x) for count data
    - "log2p": log2(1 + x)
    - "zscore": same as standard
    - "yeo-johnson": power transform (handles neg + zero)
    - "box-cox": power transform (pos only)
    - "none": no transformation
    """
    numeric_df = df.select_dtypes(include=[np.number]).copy()
    metadata = {}
    
    if method == "standard":
        scaler = StandardScaler()
        normalized = pd.DataFrame(
            scaler.fit_transform(numeric_df),
            columns=numeric_df.columns,
            index=numeric_df.index
        )
        metadata = {
            "method": "standard",
            "mean": scaler.mean_.tolist(),
            "std": scaler.scale_.tolist()
        }
    
    elif method == "robust":
        scaler = RobustScaler()
        normalized = pd.DataFrame(
            scaler.fit_transform(numeric_df),
            columns=numeric_df.columns,
            index=numeric_df.index
        )
        metadata = {
            "method": "robust",
            "median": scaler.center_.tolist(),
            "iqr": scaler.scale_.tolist()
        }
    
    elif method == "minmax":
        scaler = MinMaxScaler()
        normalized = pd.DataFrame(
            scaler.fit_transform(numeric_df),
            columns=numeric_df.columns,
            index=numeric_df.index
        )
        metadata = {
            "method": "minmax",
            "min": scaler.data_min_.tolist(),
            "max": scaler.data_max_.tolist()
        }
    
    elif method == "log1p":
        normalized = pd.DataFrame(
            np.log1p(numeric_df.clip(lower=0)),
            columns=numeric_df.columns,
            index=numeric_df.index
        )
        metadata = {"method": "log1p", "transform": "log(1+x)"}
    
    elif method == "log2p":
        normalized = pd.DataFrame(
            np.log2(numeric_df.clip(lower=1)),
            columns=numeric_df.columns,
            index=numeric_df.index
        )
        metadata = {"method": "log2p", "transform": "log2(1+x)"}
    
    elif method == "yeo-johnson":
        pt = PowerTransformer(method="yeo-johnson")
        normalized = pd.DataFrame(
            pt.fit_transform(numeric_df),
            columns=numeric_df.columns,
            index=numeric_df.index
        )
        metadata = {
            "method": "yeo-johnson",
            "lambdas": pt.lambdas_.tolist()
        }
    
    elif method == "box-cox":
        shifted = numeric_df.copy()
        shifted = shifted - shifted.min() + 0.1
        pt = PowerTransformer(method="box-cox")
        normalized = pd.DataFrame(
            pt.fit_transform(shifted),
            columns=numeric_df.columns,
            index=numeric_df.index
        )
        metadata = {
            "method": "box-cox",
            "lambdas": pt.lambdas_.tolist(),
            "note": "Data shifted to be positive"
        }
    
    else:  # "none"
        normalized = numeric_df
        metadata = {"method": "none"}
    
    return normalized, metadata

def get_normalization_recommendations(df: pd.DataFrame) -> Dict[str, str]:
    """Suggest normalization based on data distribution."""
    numeric_df = df.select_dtypes(include=[np.number])
    recommendations = {}
    
    for col in numeric_df.columns:
        series = numeric_df[col].dropna()
        if len(series) == 0:
            continue
        
        skew = stats.skew(series)
        kurtosis = stats.kurtosis(series)
        
        has_negative = (series < 0).any()
        has_zeros = (series == 0).any()
        
        if abs(skew) > 1:
            if has_negative:
                recommendations[col] = "yeo-johnson (skewed + negatives)"
            elif has_zeros:
                recommendations[col] = "log1p (count-like data)"
            else:
                recommendations[col] = "box-cox (positive skewed)"
        elif abs(kurtosis) > 2:
            recommendations[col] = "robust (heavy tails)"
        else:
            recommendations[col] = "standard (normal-like)"
    
    return recommendations

def compare_distributions(
    original: pd.Series,
    normalized: pd.Series,
    method: str
) -> Dict:
    """Compare before/after normalization."""
    return {
        "method": method,
        "original": {
            "mean": float(original.mean()),
            "std": float(original.std()),
            "min": float(original.min()),
            "max": float(original.max()),
            "skew": float(stats.skew(original.dropna())),
            "kurtosis": float(stats.kurtosis(original.dropna()))
        },
        "normalized": {
            "mean": float(normalized.mean()),
            "std": float(normalized.std()),
            "min": float(normalized.min()),
            "max": float(normalized.max()),
            "skew": float(stats.skew(normalized.dropna())),
            "kurtosis": float(stats.kurtosis(normalized.dropna()))
        }
    }

def detect_batch_effects(df: pd.DataFrame, batch_col: pd.Series) -> Dict:
    """
    Detect batch effects by comparing distributions across batches.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    
    batch_stats = {}
    for batch in batch_col.unique():
        mask = batch_col == batch
        batch_data = numeric_df[mask]
        batch_stats[str(batch)] = {
            "n_samples": len(batch_data),
            "mean_expression": batch_data.mean().mean(),
            "median_expression": batch_data.median().median(),
            "library_size_median": batch_data.sum(axis=1).median()
        }
    
    return batch_stats