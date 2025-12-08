import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer

def suggest_recipe(df: pd.DataFrame) -> dict:
    # Simple heuristic: log1p if many values > 1000; robust scaler for heavy tails
    numeric = df.select_dtypes(include=[np.number])
    log1p = (numeric.gt(1000).mean().mean() > 0.1) if not numeric.empty else False
    return {"log1p": bool(log1p), "scaler": "robust", "imputer": "median"}

def apply_recipe(df: pd.DataFrame, recipe: dict, group_col=None):
    X = df.copy()
    meta_cols = []
    if group_col and group_col in X.columns:
        meta_cols.append(group_col)
    # Separate numeric / categorical
    num = X.select_dtypes(include=[np.number])
    cat = X.select_dtypes(exclude=[np.number])

    # Impute
    if recipe.get("imputer") in {"mean", "median"} and not num.empty:
        strategy = recipe["imputer"]
        imp = SimpleImputer(strategy=strategy)
        num = pd.DataFrame(imp.fit_transform(num), columns=num.columns, index=num.index)

    # RNA-seq preset: assume counts → log1p
    if recipe.get("rnaseq"):
        num = np.log1p(num)
    else:
        if recipe.get("log1p"):
            num = np.log1p(num.clip(lower=0))

    # Scale
    scaler = recipe.get("scaler", "none")
    if scaler == "standard" and not num.empty:
        num = pd.DataFrame(StandardScaler().fit_transform(num), columns=num.columns, index=num.index)
    elif scaler == "robust" and not num.empty:
        num = pd.DataFrame(RobustScaler().fit_transform(num), columns=num.columns, index=num.index)

    # One-hot categoricals
    if recipe.get("one_hot", True) and not cat.empty:
        cat = pd.get_dummies(cat, drop_first=True)

    # Recombine
    if not num.empty and not cat.empty:
        Z = pd.concat([num, cat], axis=1)
    elif not num.empty:
        Z = num
    else:
        Z = cat

    # Meta table contains mappable columns; scores/embeddings added later
    meta = df[meta_cols].copy() if meta_cols else pd.DataFrame(index=df.index)
    return Z, meta
