import os, sys
import numpy as np
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
# Imports from core
# ------------------------------------------------------------
from core.embeddings import run_pca, run_umap, run_tsne
from core.metrics import trustworthiness_knn
from core.viz import plot_embedding

# ------------------------------------------------------------
# Page header
# ------------------------------------------------------------
st.header("3) Dimensionality Reduction")

# ------------------------------------------------------------
# Preconditions
# ------------------------------------------------------------
if "X" not in st.session_state:
    st.warning("Run a preprocess recipe first.")
    st.stop()

X = st.session_state["X"]
meta = st.session_state.get("prep_meta", pd.DataFrame(index=X.index))
raw_df = st.session_state.get("raw_df", None)
group_col = st.session_state.get("group_col", None)

# ------------------------------------------------------------
# Method selection
# ------------------------------------------------------------
method = st.radio("Method", ["PCA", "UMAP", "t-SNE"], horizontal=True)

# Metric selector (method-specific)
if method == "UMAP":
    metric = st.selectbox(
        "Distance metric (UMAP)",
        ["euclidean", "cosine", "correlation"],
        index=0
    )
elif method == "t-SNE":
    metric = st.selectbox(
        "Distance metric (t-SNE)",
        ["euclidean", "cosine"],
        index=0
    )
else:
    metric = None

# ============================================================
# PCA
# ============================================================
if method == "PCA":

    n_comp = st.slider(
        "Components",
        2,
        max(2, min(10, X.shape[1])),
        2
    )

    pca_res = run_pca(X, n_components=n_comp)

    # Build metadata for plotting
    out = pca_res.scores.copy()

    if raw_df is not None and group_col in raw_df.columns:
        out[group_col] = raw_df[group_col].values

    if raw_df is not None and "Replicate" in raw_df.columns:
        out["Replicate"] = raw_df["Replicate"].values

    st.session_state["prep_meta"] = out

    # Plot PCA (ggplot-like)
    fig = plot_embedding(
        out,
        x="PC1",
        y="PC2",
        color=group_col,
        symbol="Replicate" if "Replicate" in out.columns else None,
        title=f"PCA projection | n_components={n_comp}",
    )

    st.plotly_chart(fig, use_container_width=True)
    st.plotly_chart(pca_res.fig_scree, use_container_width=True)
    st.dataframe(pca_res.loadings.head(20), use_container_width=True)

# ============================================================
# UMAP / t-SNE
# ============================================================
else:

    # ---------------- Parameters ----------------
    if method == "UMAP":
        n_neighbors = st.slider("n_neighbors", 5, 50, 15)
        min_dist = st.slider("min_dist", 0.0, 0.99, 0.1)

        emb = run_umap(
            X,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
        )

        coords = pd.DataFrame(
            emb.embedding,
            columns=["UMAP1", "UMAP2"],
            index=X.index
        )

        xcol, ycol = "UMAP1", "UMAP2"
        title = f"UMAP | n_neighbors={n_neighbors}, min_dist={min_dist}, metric={metric}"

    else:  # t-SNE
        perplexity = st.slider(
            "perplexity (t-SNE)",
            5,
            min(80, max(5, X.shape[0] // 3)),
            30
        )

        emb = run_tsne(
            X,
            perplexity=perplexity,
            metric=metric
        )

        coords = pd.DataFrame(
            emb.embedding,
            columns=["TSNE1", "TSNE2"],
            index=X.index
        )

        xcol, ycol = "TSNE1", "TSNE2"
        title = f"t-SNE | perplexity={perplexity}, metric={metric}"

    # ---------------- Merge with metadata ----------------
    out = meta.copy()
    for c in coords.columns:
        out[c] = coords[c]

    if raw_df is not None and group_col in raw_df.columns:
        out[group_col] = raw_df[group_col].values

    if raw_df is not None and "Replicate" in raw_df.columns:
        out["Replicate"] = raw_df["Replicate"].values

    st.session_state["prep_meta"] = out

    # ---------------- Quality metrics ----------------
    from sklearn.decomposition import PCA

    if X.shape[0] > 5:
        Xp = PCA(n_components=0.9).fit_transform(X.values)
        tw, knn_pres = trustworthiness_knn(Xp, emb.embedding)
    else:
        tw, knn_pres = np.nan, np.nan

    m1, m2 = st.columns(2)
    m1.metric("Trustworthiness", f"{tw:.3f}")
    m2.metric("KNN preservation", f"{knn_pres:.3f}")

    # ---------------- Plot embedding ----------------
    fig = plot_embedding(
        out,
        x=xcol,
        y=ycol,
        color=group_col,
        symbol="Replicate" if "Replicate" in out.columns else None,
        title=title,
    )

    st.plotly_chart(fig, use_container_width=True)

    st.info("Next → Visualize Studio for color / shape / facet mapping")
