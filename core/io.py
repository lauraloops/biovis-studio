import pandas as pd
import numpy as np
from typing import Tuple, Dict, List

def read_any(file):
    """Read CSV/TSV/Parquet with flexible parsing."""
    name = getattr(file, "name", "uploaded")
    if name.endswith(".parquet"):
        return pd.read_parquet(file)
    if name.endswith(".tsv") or name.endswith(".txt"):
        return pd.read_csv(file, sep="\t")
    return pd.read_csv(file)

def infer_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze data types, missing values, uniqueness."""
    parts = []
    for c in df.columns:
        s = df[c]
        dtype = str(s.dtype)
        na_pct = float(s.isna().mean() * 100)
        uniq = int(s.nunique(dropna=True))
        is_numeric = pd.api.types.is_numeric_dtype(s)
        parts.append({
            "column": c,
            "dtype": dtype,
            "missing_%": round(na_pct, 2),
            "unique_vals": uniq,
            "is_numeric": is_numeric
        })
    return pd.DataFrame(parts)

def detect_id_columns(df: pd.DataFrame, threshold: float = 0.95) -> List[str]:
    """
    Identify likely ID columns:
    - Very high cardinality (unique_count ≈ row_count)
    - Non-numeric or string-like
    - Low variance if numeric
    """
    candidates = []
    n_rows = len(df)
    for col in df.columns:
        s = df[col]
        uniq = s.nunique(dropna=True)
        uniq_ratio = uniq / n_rows
        is_numeric = pd.api.types.is_numeric_dtype(s)
        
        if uniq_ratio > threshold:
            candidates.append({
                "column": col,
                "cardinality_ratio": round(uniq_ratio, 3),
                "is_numeric": is_numeric,
                "likely_id": True
            })
    return [c["column"] for c in candidates]

def detect_duplicate_ids(df: pd.DataFrame) -> Dict[str, int]:
    """Detect if there are duplicate ID columns (same data, different names)."""
    numeric_df = df.select_dtypes(include=[np.number])
    duplicates = {}
    
    for i, col1 in enumerate(numeric_df.columns):
        for col2 in numeric_df.columns[i+1:]:
            if numeric_df[col1].equals(numeric_df[col2]):
                duplicates[f"{col1}_{col2}"] = "identical"
            elif numeric_df[col1].corr(numeric_df[col2]) > 0.99:
                duplicates[f"{col1}_{col2}"] = "highly_correlated"
    
    return duplicates

def infer_orientation(df: pd.DataFrame) -> Dict[str, str]:
    """
    Detect if data is samples×features or features×samples.
    Returns: {"orientation": "samples_x_features" | "features_x_samples",
              "reason": "explanation", "recommendation": "..."}
    """
    n_rows, n_cols = df.shape
    numeric_cols = df.select_dtypes(include=[np.number]).shape[1]
    
    if n_cols > n_rows * 2:
        return {
            "orientation": "features_x_samples",
            "reason": f"More columns ({n_cols}) than rows ({n_rows}). Likely features × samples.",
            "recommendation": "Consider transposing."
        }
    else:
        return {
            "orientation": "samples_x_features",
            "reason": f"More rows ({n_rows}) than columns ({n_cols}). Standard format.",
            "recommendation": "Keep as is."
        }

def transpose_option(df: pd.DataFrame) -> pd.DataFrame:
    """Transpose DataFrame while preserving index."""
    return df.T

def remove_columns(df: pd.DataFrame, cols_to_drop: List[str]) -> pd.DataFrame:
    """Remove specified columns."""
    return df.drop(columns=cols_to_drop, errors='ignore')

def identify_metadata_numeric_separation(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Separate metadata (non-numeric) from expression/feature data (numeric).
    Returns: (metadata_df, features_df)
    """
    numeric_df = df.select_dtypes(include=[np.number])
    metadata_df = df.select_dtypes(exclude=[np.number])
    return metadata_df, numeric_df
