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
from core.preprocess import apply_recipe, suggest_recipe  # noqa: E401

st.header("2) Preprocess Recipe")

if "raw_df" not in st.session_state:
    st.warning("Go to Import & Profile first.")
    st.stop()

raw = st.session_state["raw_df"]

# Auto-suggest a recipe based on schema
suggestion = suggest_recipe(raw)
with st.expander("Suggested recipe (click to open)"):
    st.json(suggestion, expanded=False)

st.subheader("Edit Recipe")
with st.form("recipe_form"):
    log1p = st.checkbox("Apply log1p to positive-skewed numeric columns", value=suggestion["log1p"]) 
    scale = st.selectbox("Scaling", ["none", "standard", "robust"], index=["none","standard","robust"].index(suggestion["scaler"]))
    impute = st.selectbox("Imputation", ["none", "median", "mean"], index=["none","median","mean"].index(suggestion["imputer"]))
    one_hot = st.checkbox("One-hot encode categoricals", value=True)
    is_rnaseq = st.checkbox("RNA-seq quick preset (TPM/CPM→log1p)", value=False)
    submitted = st.form_submit_button("Apply")

if submitted:
    recipe = {
        "log1p": log1p or is_rnaseq,
        "scaler": scale,
        "imputer": impute,
        "one_hot": one_hot,
        "rnaseq": is_rnaseq,
    }
    X, meta = apply_recipe(raw, recipe, group_col=st.session_state.get("group_col"))
    st.session_state["X"] = X
    st.session_state["prep_meta"] = meta
    st.success(f"Preprocessed matrix: {X.shape}")
    st.info("Next → Dimensionality Reduction (sidebar).")
