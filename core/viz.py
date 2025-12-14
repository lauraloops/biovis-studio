import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import chi2

# ------------------------------------------------------------
# Confidence ellipse helper
# ------------------------------------------------------------
def _confidence_ellipse(x, y, level=0.95):
    cov = np.cov(x, y)
    mean = np.mean(x), np.mean(y)

    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]

    theta = np.linspace(0, 2 * np.pi, 200)
    circle = np.vstack((np.cos(theta), np.sin(theta)))

    scale = np.sqrt(chi2.ppf(level, 2))
    ellipse = eigvecs @ np.diag(np.sqrt(eigvals)) @ circle * scale

    return ellipse[0] + mean[0], ellipse[1] + mean[1]

# ------------------------------------------------------------
# Main plotting function
# ------------------------------------------------------------
def plot_embedding(
    df: pd.DataFrame,
    x: str,
    y: str,
    color: str | None = None,
    symbol: str | None = None,
    size: str | None = None,
    title: str | None = None,
    add_ellipses: bool = True,
    axis_labels: tuple[str, str] | None = None,
):
    """
    Publication-style interactive embedding plot (PCA / UMAP / t-SNE)
    """

    fig = px.scatter(
        df,
        x=x,
        y=y,
        color=color,
        symbol=symbol,
        size=size,
        hover_data=df.columns,
        opacity=0.85,
    )

    fig.update_traces(
        marker=dict(
            line=dict(width=0.6, color="black"),
            sizemode="area",
            sizeref=0.05,
        )
    )

    fig.update_layout(
        title=title,
        template="simple_white",
        font=dict(size=14),
        legend_title_text="",
        margin=dict(l=20, r=20, t=60, b=20),
    )

    if axis_labels:
        fig.update_xaxes(title=axis_labels[0])
        fig.update_yaxes(title=axis_labels[1])

    # --------------------------------------------------------
    # Optional confidence ellipses (ggplot-style)
    # --------------------------------------------------------
    if add_ellipses and color is not None:
        for group, gdf in df.groupby(color):
            if gdf.shape[0] < 5:
                continue
            ex, ey = _confidence_ellipse(gdf[x], gdf[y])
            fig.add_trace(
                go.Scatter(
                    x=ex,
                    y=ey,
                    mode="lines",
                    line=dict(width=2),
                    name=f"{group} ellipse",
                    showlegend=False,
                )
            )

    return fig
