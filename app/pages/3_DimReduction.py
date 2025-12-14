import os, sys
import numpy as np
import pandas as pd
import streamlit as st

# ------------------------------------------------------------
# Path fix so `core/` is importable
# ------------------------------------------------------------
HERE = os.path.dirname(__file__)
for ROOT in [
    os.path.abspath(os.path.join(HERE, "..")),
    os.path.abspath(os.path.join(HERE, "..", "..")),
]:
    if os.path.isdir(os.path.join(ROOT, "core")) and ROOT not in sys.path:
        sys.path.insert(0, ROOT)
        break

# ------------------------------------------------------------
# Imports
# ------------------------------------------------------------
from core.embeddings import run_pca, run_umap, run_tsne
from core.metrics import trustworthiness_knn
from core.viz import plot_embedding

# ------------------------------------------------------------
# Page
# ------------------------------------------------------------
st.header("3) Dimensionality Reduction")

if "X" not in st.session_state:
    st.warning("Run a preprocess recipe first.")
    st.stop()

X = st.session_state["X"]
raw_df = st.session_state.get("raw_df")
group_col = st.session_state.get("group_col")

method = st.radio("Method", ["PCA", "UMAP", "t-SNE"], horizontal=True)

# ------------------------------------------------------------
# PCA
# ------------------------------------------------------------
if method == "PCA":

    n_comp = st.slider("Components", 2, min(10, X.shape[1]), 2)
    pca_res = run_pca(X, n_components=n_comp)

    out = pca_res.scores.copy()

    if raw_df is not None and group_col in raw_df.columns:
        out[group_col] = raw_df[group_col].values

    st.session_state["prep_meta"] = out

    fig = plot_embedding(
        out,
        x="PC1",
        y="PC2",
        color=group_col,
        title=f"PCA projection | n_components={n_comp}",
        axis_labels=("PC1", "PC2"),
        add_ellipses=True,
    )

    st.plotly_chart(fig, use_container_width=True)
    st.plotly_chart(pca_res.fig_scree, use_container_width=True)
    st.dataframe(pca_res.loadings.head(20), use_container_width=True)

# ------------------------------------------------------------
# UMAP / t-SNE
# ------------------------------------------------------------
else:

    if method == "UMAP":
        metric = st.selectbox("Metric", ["euclidean", "cosine"])
        n_neighbors = st.slider("n_neighbors", 5, 50, 15)
        min_dist = st.slider("min_dist", 0.0, 0.99, 0.1)

        emb = run_umap(X, n_neighbors=n_neighbors, min_dist=min_dist, metric=metric)
        coords = pd.DataFrame(emb.embedding, columns=["UMAP1", "UMAP2"], index=X.index)
        xcol, ycol = "UMAP1", "UMAP2"

    else:
        metric = st.selectbox("Metric", ["euclidean", "cosine"])
        perplexity = st.slider("perplexity", 5, min(80, X.shape[0] // 3), 30)

        emb = run_tsne(X, perplexity=perplexity, metric=metric)
        coords = pd.DataFrame(emb.embedding, columns=["TSNE1", "TSNE2"], index=X.index)
        xcol, ycol = "TSNE1", "TSNE2"

    out = coords.copy()

    if raw_df is not None and group_col in raw_df.columns:
        out[group_col] = raw_df[group_col].values

    st.session_state["prep_meta"] = out

    from sklearn.decomposition import PCA
    Xp = PCA(n_components=0.9).fit_transform(X.values)
    tw, knn = trustworthiness_knn(Xp, emb.embedding)

    st.metric("Trustworthiness", f"{tw:.3f}")
    st.metric("KNN preservation", f"{knn:.3f}")

    fig = plot_embedding(
        out,
        x=xcol,
        y=ycol,
        color=group_col,
        title=f"{method} projection",
        add_ellipses=True,
    )

    st.plotly_chart(fig, use_container_width=True)
