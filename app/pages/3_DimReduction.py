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
from core.dr import run_pca, run_umap, run_tsne
from core.metrics import trustworthiness_knn

st.header("3) Dimensionality Reduction")

if "X" not in st.session_state:
    st.warning("Run a preprocess recipe first.")
    st.stop()

X = st.session_state["X"]
meta = st.session_state.get("prep_meta", pd.DataFrame(index=X.index))

method = st.radio("Method", ["PCA", "UMAP", "t-SNE"], horizontal=True)
# ---------------- Metric selection ----------------
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
# ---------------- Run DR ----------------
if method == "PCA":
    n_comp = st.slider("Components", 2, max(2, min(10, X.shape[1])), 2)
    pca_res = run_pca(X, n_components=n_comp)

    # Combine group info if present
    group_col = st.session_state.get("group_col")
    if group_col and group_col in st.session_state["raw_df"].columns:
        merged = pca_res.scores.copy()
        merged[group_col] = st.session_state["raw_df"][group_col].values
        st.session_state["prep_meta"] = merged
    else:
        st.session_state["prep_meta"] = pca_res.scores.copy()

    st.plotly_chart(pca_res.fig_scores, use_container_width=True)
    st.plotly_chart(pca_res.fig_scree, use_container_width=True)
    st.dataframe(pca_res.loadings.head(20), use_container_width=True)

else:
    if method == "UMAP":
        n_neighbors = st.slider("n_neighbors", 5, 50, 15)
        min_dist = st.slider("min_dist", 0.0, 0.99, 0.1)
        emb = run_umap(X,n_neighbors=n_neighbors,min_dist=min_dist,metric=metric,)

        coords = pd.DataFrame(emb.embedding, columns=["UMAP1","UMAP2"], index=X.index)
    else:
        perplexity = st.slider("perplexity (t-SNE)", 5, min(80, max(5, X.shape[0]//3)), 30)
        emb = run_tsne(X, perplexity=perplexity, metric=metric)
        coords = pd.DataFrame(emb.embedding, columns=["TSNE1","TSNE2"], index=X.index)

    # Merge coords with any existing meta (group, etc.)
    out = meta.copy()
    for c in coords.columns: out[c] = coords[c]
    st.session_state["prep_meta"] = out

    from sklearn.decomposition import PCA

    Xp = PCA(n_components=0.9).fit_transform(X.values)
    tw, knn_pres = trustworthiness_knn(Xp, emb.embedding)

    m1, m2 = st.columns(2)
    m1.metric("Trustworthiness", f"{tw:.3f}")
    m2.metric("KNN preservation", f"{knn_pres:.3f}")
    st.plotly_chart(emb.fig_scatter, use_container_width=True)

    st.info("Next → Visualize Studio for color/shape/facet mapping")
