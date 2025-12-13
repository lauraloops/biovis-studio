from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
import umap
from openTSNE import TSNE


# ============================================================
# Result containers (KEEP THESE)
# ============================================================
@dataclass
class PCAResult:
    scores: pd.DataFrame
    loadings: pd.DataFrame
    fig_scores: go.Figure
    fig_scree: go.Figure


@dataclass
class EmbeddingResult:
    embedding: np.ndarray
    fig_scatter: go.Figure


# ============================================================
# PCA (deterministic, user-controlled)
# ============================================================
def run_pca(X: pd.DataFrame, n_components: int | float = 2) -> PCAResult:
    pca = PCA(n_components=n_components)
    comps = pca.fit_transform(X)

    n_final = comps.shape[1]
    cols = [f"PC{i+1}" for i in range(n_final)]

    scores = pd.DataFrame(comps, columns=cols, index=X.index)
    loadings = pd.DataFrame(pca.components_.T, index=X.columns, columns=cols)

    fig_scores = px.scatter(
        scores, x=cols[0], y=cols[1], title="PCA Scores"
    )

    evr = pca.explained_variance_ratio_
    fig_scree = go.Figure(go.Bar(x=cols, y=evr))
    fig_scree.update_layout(title="Explained Variance Ratio")

    return PCAResult(scores, loadings, fig_scores, fig_scree)


# ============================================================
# Internal PCA preprocessing (NEW)
# ============================================================
def _pca_preprocess(X: pd.DataFrame, n_components: int | float):
    pca = PCA(n_components=n_components)
    Xp = pca.fit_transform(X)
    return pd.DataFrame(
        Xp,
        index=X.index,
        columns=[f"PC{i+1}" for i in range(Xp.shape[1])]
    )


# ============================================================
# Automatic parameter heuristics (NEW)
# ============================================================
def _auto_tsne_params(n_samples, perplexity=None, learning_rate=None):
    if perplexity is None:
        p = int(np.sqrt(n_samples))
        p = max(5, min(p, min(50, n_samples // 3)))
        perplexity = p
    if learning_rate is None:
        learning_rate = max(200, n_samples / 12)
    return perplexity, learning_rate


def _auto_umap_params(n_samples, n_neighbors=None, min_dist=None):
    if n_neighbors is None:
        n_neighbors = max(5, min(50, int(np.log10(max(n_samples, 10)) * 10)))
    if min_dist is None:
        min_dist = 0.1
    return n_neighbors, min_dist


# ============================================================
# UMAP (UPGRADED, same API)
# ============================================================
def run_umap(
    X: pd.DataFrame,
    n_neighbors: int | None = None,
    min_dist: float | None = None,
    metric: str = "euclidean",
    use_pca: bool = True,
    pca_components: int | float = 0.9,
    n_components: int = 2,
) -> EmbeddingResult:

    X_used = _pca_preprocess(X, pca_components) if use_pca else X
    n_neighbors, min_dist = _auto_umap_params(
        X_used.shape[0], n_neighbors, min_dist
    )

    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric=metric,
        random_state=0,
    )

    emb = reducer.fit_transform(X_used)

    fig = px.scatter(
        x=emb[:, 0],
        y=emb[:, 1],
        title=f"UMAP (neighbors={n_neighbors}, min_dist={min_dist})",
    )

    return EmbeddingResult(emb, fig)


# ============================================================
# t-SNE (UPGRADED, same API)
# ============================================================
def run_tsne(
    X: pd.DataFrame,
    perplexity: int | None = None,
    n_iter: int = 1000,
    metric: str = "euclidean",
    use_pca: bool = True,
    pca_components: int | float = 0.9,
    n_components: int = 2,
) -> EmbeddingResult:

    X_used = _pca_preprocess(X, pca_components) if use_pca else X
    perplexity, lr = _auto_tsne_params(X_used.shape[0], perplexity)

    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        learning_rate=lr,
        n_iter=n_iter,
        initialization="pca" if use_pca else "random",
        metric=metric,
        random_state=0,
    )

    emb = np.asarray(tsne.fit(X_used))

    fig = px.scatter(
        x=emb[:, 0],
        y=emb[:, 1],
        title=f"t-SNE (perplexity={perplexity}, iter={n_iter})",
    )

    return EmbeddingResult(emb, fig)
