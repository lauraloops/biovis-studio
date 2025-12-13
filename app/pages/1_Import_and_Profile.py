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

import pandas as pd
import streamlit as st
from core.io import read_any, infer_schema

st.header("1) Import & Profile")

uploaded = st.file_uploader("Upload CSV / TSV / Parquet", type=["csv", "tsv", "txt", "parquet"]) 

if uploaded:
    df = read_any(uploaded)
    st.session_state["raw_df"] = df
    st.success(f"Loaded shape: {df.shape[0]} rows × {df.shape[1]} cols")

    schema = infer_schema(df)
    st.subheader("Schema & Quality")
    st.dataframe(schema, use_container_width=True)

    st.subheader("Preview")
    st.dataframe(df.head(20), use_container_width=True)

    st.subheader("Column Role Mapping")
    cols = list(df.columns)
    group_col = st.selectbox("Group column (optional)", [None] + cols, index=0)
    batch_col = st.selectbox("Batch column (optional)", [None] + cols, index=0)
    st.session_state["group_col"] = group_col
    st.session_state["batch_col"] = batch_col

    st.info("Next → Preprocess recipe (sidebar).")
else:
    st.info("Upload a file to begin. A small demo is available in /data.")
