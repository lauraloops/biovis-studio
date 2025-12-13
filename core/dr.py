from dataclasses import dataclass
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
import umap
from openTSNE import TSNE

@dataclass
class PCAResult:
    scores: pd.DataFrame
    loadings: pd.DataFrame
    fig_scores: go.Figure
    fig_scree: go.Figure

def run_pca(X: pd.DataFrame, n_components=2) -> PCAResult:
    pca = PCA(n_components=n_components, random_state=0)
    comps = pca.fit_transform(X)
    cols = [f"PC{i+1}" for i in range(n_components)]
    scores = pd.DataFrame(comps, columns=cols, index=X.index)
    loadings = pd.DataFrame(pca.components_.T, index=X.columns, columns=cols)

    fig_scores = px.scatter(scores, x="PC1", y="PC2", title="PCA Scores")

    evr = pca.explained_variance_ratio_
    fig_scree = go.Figure(go.Bar(x=[f"PC{i+1}" for i in range(len(evr))], y=evr))
    fig_scree.update_layout(title="Explained Variance Ratio")

    return PCAResult(scores, loadings, fig_scores, fig_scree)

@dataclass
class EmbeddingResult:
    embedding: np.ndarray
    fig_scatter: go.Figure

def run_umap(X: pd.DataFrame, n_neighbors=15, min_dist=0.1) -> EmbeddingResult:
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=0)
    emb = reducer.fit_transform(X)
    fig = px.scatter(x=emb[:,0], y=emb[:,1], title="UMAP")
    return EmbeddingResult(emb, fig)

def run_tsne(X: pd.DataFrame, perplexity=30) -> EmbeddingResult:
    emb = TSNE(perplexity=perplexity, random_state=0).fit(X)
    emb_arr = np.asarray(emb)  # openTSNE Embedding -> ndarray
    fig = px.scatter(x=emb_arr[:,0], y=emb_arr[:,1], title="t-SNE")
    return EmbeddingResult(emb_arr, fig)
