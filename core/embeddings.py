from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
import umap
from openTSNE import TSNE


# ============================================================
# Result containers
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
    """
    Deterministic PCA.
    n_components:
      - int  -> number of PCs
      - float -> fraction of variance to keep (0 < n_components < 1)
    """
    pca = PCA(n_components=n_components)
    comps = pca.fit_transform(X)

    n_final = comps.shape[1]
    cols = [f"PC{i+1}" for i in range(n_final)]

    scores = pd.DataFrame(comps, columns=cols, index=X.index)
    loadings = pd.DataFrame(pca.components_.T, index=X.columns, columns=cols)

    # Scores plot
    if n_final >= 2:
        fig_scores = px.scatter(scores, x="PC1", y="PC2", title="PCA Scores")
    else:
        fig_scores = px.scatter(
            x=scores["PC1"],
            y=np.zeros(len(scores)),
            title="PCA Scores (1D)"
        )

    # Scree plot (explained + cumulative)
    evr = pca.explained_variance_ratio_
    cum_evr = np.cumsum(evr)

    fig_scree = go.Figure()
    fig_scree.add_bar(x=cols, y=evr, name="Explained variance")
    fig_scree.add_scatter(
        x=cols, y=cum_evr,
        name="Cumulative",
        yaxis="y2"
    )

    fig_scree.update_layout(
        title="Explained Variance Ratio",
        yaxis=dict(title="Explained variance"),
        yaxis2=dict(
            title="Cumulative variance",
            overlaying="y",
            side="right"
        ),
        legend=dict(orientation="h")
    )

    return PCAResult(scores, loadings, fig_scores, fig_scree)


def pca_preprocess(X: pd.DataFrame, n_components: int | float = 0.9) -> pd.DataFrame:
    """
    PCA preprocessing for UMAP / t-SNE.
    """
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    cols = [f"PC{i+1}" for i in range(X_pca.shape[1])]
    return pd.DataFrame(X_pca, columns=cols, index=X.index)


# ============================================================
# Automatic parameter heuristics
# ============================================================
def auto_tsne_params(
    n_samples: int,
    perplexity: int | None = None,
    learning_rate: float | None = None
) -> dict:
    """
    Automatic t-SNE heuristics if user does not provide parameters.
    """
    if perplexity is None:
        p = int(np.sqrt(n_samples))
        p = max(5, min(p, min(50, n_samples // 3)))
        perplexity = p

    if learning_rate is None:
        learning_rate = max(200, n_samples / 12)

    return {
        "perplexity": perplexity,
        "learning_rate": learning_rate
    }


def auto_umap_params(
    n_samples: int,
    n_neighbors: int | None = None,
    min_dist: float | None = None
) -> dict:
    """
    Automatic UMAP heuristics if user does not provide parameters.
    """
    if n_neighbors is None:
        nn = int(np.log10(max(n_samples, 10)) * 10)
        nn = max(5, min(nn, 50))
        n_neighbors = nn

    if min_dist is None:
        min_dist = 0.1

    return {
        "n_neighbors": n_neighbors,
        "min_dist": min_dist
    }


# ============================================================
# UMAP (optional PCA chaining)
# ============================================================
def run_umap(
    X: pd.DataFrame,
    use_pca: bool = True,
    pca_components: int | float = 0.9,
    n_components: int = 2,
    n_neighbors: int | None = None,
    min_dist: float | None = None,
    metric: str = "euclidean"
) -> EmbeddingResult:

    X_used = pca_preprocess(X, pca_components) if use_pca else X

    auto = auto_umap_params(
        n_samples=X_used.shape[0],
        n_neighbors=n_neighbors,
        min_dist=min_dist
    )

    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=auto["n_neighbors"],
        min_dist=auto["min_dist"],
        metric=metric,
        random_state=0
    )

    emb = reducer.fit_transform(X_used)

    fig = px.scatter(
        x=emb[:, 0],
        y=emb[:, 1],
        title=(
            f"UMAP | PCA={use_pca}, PCs={pca_components}, "
            f"neighbors={auto['n_neighbors']}, "
            f"min_dist={auto['min_dist']}, "
            f"metric={metric}"
        )
    )

    return EmbeddingResult(emb, fig)


# ============================================================
# t-SNE (openTSNE, optional PCA chaining)
# ============================================================
def run_tsne(
    X: pd.DataFrame,
    use_pca: bool = True,
    pca_components: int | float = 0.9,
    n_components: int = 2,
    n_iter: int = 1000,
    perplexity: int | None = None,
    learning_rate: float | None = None,
    metric: str = "euclidean"
) -> EmbeddingResult:

    X_used = pca_preprocess(X, pca_components) if use_pca else X
    init = "pca" if use_pca else "random"

    auto = auto_tsne_params(
        n_samples=X_used.shape[0],
        perplexity=perplexity,
        learning_rate=learning_rate
    )

    tsne = TSNE(
        n_components=n_components,
        perplexity=auto["perplexity"],
        learning_rate=auto["learning_rate"],
        n_iter=n_iter,
        initialization=init,
        metric=metric,
        random_state=0
    )

    emb = tsne.fit(X_used)
    emb_arr = np.asarray(emb)

    fig = px.scatter(
        x=emb_arr[:, 0],
        y=emb_arr[:, 1],
        title=(
            f"t-SNE | PCA={use_pca}, PCs={pca_components}, "
            f"perplexity={auto['perplexity']}, "
            f"lr={auto['learning_rate']}, "
            f"n_iter={n_iter}"
        )
    )

    return EmbeddingResult(emb_arr, fig)