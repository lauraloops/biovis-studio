import os, sys
HERE = os.path.dirname(__file__)
CANDIDATES = [
    os.path.abspath(os.path.join(HERE, "..")),        # works for Home.py
    os.path.abspath(os.path.join(HERE, "..", "..")),  # works for pages/*
]
for ROOT in CANDIDATES:
    if os.path.isdir(os.path.join(ROOT, "core")) and ROOT not in sys.path:
        sys.path.insert(0, ROOT)
        break

import streamlit as st
import pandas as pd
import numpy as np
from core.preprocess import apply_recipe, suggest_recipe  # noqa: E401

st.header("2) Preprocess Recipe")

# ============================================================
# Guards
# ============================================================
if "raw_df" not in st.session_state:
    st.warning("Go to Import & Profile first.")
    st.stop()

raw = st.session_state["raw_df"]

# ============================================================
# Suggested recipe
# ============================================================
suggestion = suggest_recipe(raw)
with st.expander("Suggested recipe (click to open)"):
    st.json(suggestion, expanded=False)

# ============================================================
# Recipe editor
# ============================================================
st.subheader("Edit Recipe")

with st.form("recipe_form"):

    # ---- Core preprocessing ----
    log1p = st.checkbox(
        "Apply log1p to positive-skewed numeric columns",
        value=suggestion["log1p"]
    )

    scale = st.selectbox(
        "Scaling",
        ["none", "standard", "robust"],
        index=["none", "standard", "robust"].index(suggestion["scaler"])
    )

    impute = st.selectbox(
        "Imputation",
        ["none", "median", "mean"],
        index=["none", "median", "mean"].index(suggestion["imputer"])
    )

    one_hot = st.checkbox(
        "One-hot encode categoricals",
        value=True
    )

    is_rnaseq = st.checkbox(
        "RNA-seq quick preset (TPM/CPM → log1p)",
        value=False
    )

    # ---- DR-specific controls (NEW) ----
    st.markdown("### Dimensionality Reduction options")

    dr_ready = st.checkbox(
        "Optimize for dimensionality reduction (recommended)",
        value=True
    )

    var_filter = st.checkbox(
        "Remove low-variance features",
        value=True
    )

    var_quantile = st.slider(
        "Variance filter strength (drop lowest X%)",
        min_value=0,
        max_value=50,
        value=10,
        step=5
    )

    submitted = st.form_submit_button("Apply")

# ============================================================
# Apply recipe
# ============================================================
if submitted:

    # Enforce DR-safe defaults if requested
    final_log1p = log1p or is_rnaseq
    final_scaler = scale
    if dr_ready and final_scaler == "none":
        final_scaler = "standard"

    recipe = {
        "log1p": final_log1p,
        "scaler": final_scaler,
        "imputer": impute,
        "one_hot": one_hot,
        "rnaseq": is_rnaseq,
    }

    # ---- Apply core preprocessing ----
    X, meta = apply_recipe(
        raw,
        recipe,
        group_col=st.session_state.get("group_col")
    )

    # ---- Low-variance filtering (NEW) ----
    if var_filter:
        variances = X.var(axis=0)
        threshold = np.percentile(variances, var_quantile)
        keep_cols = variances[variances > threshold].index
        X = X[keep_cols]

        st.info(
            f"Low-variance filtering: kept {X.shape[1]} features "
            f"(dropped {var_quantile}%)"
        )

    # ---- Save to session ----
    st.session_state["X"] = X
    st.session_state["prep_meta"] = meta

    # ---- Feedback ----
    st.success(f"Preprocessed matrix ready for DR: {X.shape[0]} samples × {X.shape[1]} features")

    st.caption(
        "Preprocessing optimized for distance-based methods "
        "(scaling + variance filtering)."
    )

    st.info("Next → Dimensionality Reduction (sidebar).")
