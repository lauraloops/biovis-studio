import os, sys
import pandas as pd
import streamlit as st

# ------------------------------------------------------------
# Path fix so `core/` is importable
# ------------------------------------------------------------
HERE = os.path.dirname(__file__)
CANDIDATES = [
    os.path.abspath(os.path.join(HERE, "..")),
    os.path.abspath(os.path.join(HERE, "..", "..")),
]
for ROOT in CANDIDATES:
    if os.path.isdir(os.path.join(ROOT, "core")) and ROOT not in sys.path:
        sys.path.insert(0, ROOT)
        break

# ------------------------------------------------------------
# Imports
# ------------------------------------------------------------
from core.viz import plot_embedding

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def infer_axes(df):
    """Infer embedding axes from dataframe columns."""
    for pair in [("UMAP1", "UMAP2"), ("TSNE1", "TSNE2"), ("PC1", "PC2")]:
        if all(c in df.columns for c in pair):
            return pair
    raise ValueError("No embedding coordinates found (PC1/2, UMAP1/2, TSNE1/2)")

# ------------------------------------------------------------
# Page
# ------------------------------------------------------------
st.header("4) Visualize Studio")

if "prep_meta" not in st.session_state:
    st.warning("Run Dimensionality Reduction first.")
    st.stop()

meta = st.session_state["prep_meta"]

# ------------------------------------------------------------
# Infer axes
# ------------------------------------------------------------
x_col, y_col = infer_axes(meta)

# ------------------------------------------------------------
# UI controls
# ------------------------------------------------------------
cols = list(meta.columns)

color_by = st.selectbox(
    "Color by",
    [None] + cols,
    index=0
)

shape_by = st.selectbox(
    "Shape by",
    [None] + cols,
    index=0
)

numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(meta[c])]
size_by = st.selectbox(
    "Size by (numeric)",
    [None] + numeric_cols,
    index=0
)

# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------
fig = plot_embedding(
    meta,
    x=x_col,
    y=y_col,
    color=color_by if color_by is not None else None,
    symbol=shape_by if shape_by is not None else None,
    size=size_by if size_by is not None else None,
    title="Visualization Studio",
    add_ellipses=False,  # studio = aesthetics only
)

st.plotly_chart(fig, use_container_width=True)
