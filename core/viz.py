import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import chi2


# ------------------------------------------------------------
# Confidence ellipse (ggplot2::stat_ellipse equivalent)
# ------------------------------------------------------------
def _confidence_ellipse(x, y, level=0.95, n_points=200):
    """
    Compute confidence ellipse coordinates for x/y data.
    Returns arrays (ex, ey).
    """
    x = np.asarray(x)
    y = np.asarray(y)

    cov = np.cov(x, y)
    mean = np.mean(x), np.mean(y)

    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]

    theta = np.linspace(0, 2 * np.pi, n_points)
    circle = np.vstack([np.cos(theta), np.sin(theta)])

    scale = np.sqrt(chi2.ppf(level, df=2))
    ellipse = scale * eigvecs @ np.diag(np.sqrt(eigvals)) @ circle

    return ellipse[0] + mean[0], ellipse[1] + mean[1]


# ------------------------------------------------------------
# Publication-style embedding plot
# ------------------------------------------------------------
def plot_embedding(
    df: pd.DataFrame,
    x: str,
    y: str,
    color: str | None = None,
    symbol: str | None = None,
    title: str | None = None,
    axis_labels: tuple[str, str] | None = None,
    add_ellipses: bool = True,
    ellipse_level: float = 0.95,
):
    """
    PCA / UMAP / t-SNE scatter plot with optional confidence ellipses.
    """

    fig = px.scatter(
        df,
        x=x,
        y=y,
        color=color,
        symbol=symbol,
        hover_data=df.columns,
        opacity=0.9,
    )

    # Marker styling (publication-like)
    fig.update_traces(
        marker=dict(
            size=8,
            line=dict(width=0.6, color="black"),
        )
    )

    # Ax encouraging PCA interpretation
    fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.4)
    fig.add_vline(x=0, line_dash="dash", line_color="black", opacity=0.4)

    # Confidence ellipses per group
    if add_ellipses and color is not None:
        for grp, sub in df.groupby(color):
            if sub.shape[0] < 5:
                continue

            ex, ey = _confidence_ellipse(
                sub[x], sub[y], level=ellipse_level
            )

            fig.add_trace(
                go.Scatter(
                    x=ex,
                    y=ey,
                    mode="lines",
                    line=dict(width=2),
                    fill="toself",
                    opacity=0.2,
                    name=f"{grp} ({int(ellipse_level*100)}%)",
                    showlegend=True,
                )
            )

    # Layout
    fig.update_layout(
        title=title,
        template="simple_white",
        legend_title_text="",
        font=dict(size=14),
        margin=dict(l=30, r=30, t=60, b=30),
    )

    if axis_labels:
        fig.update_xaxes(title=axis_labels[0])
        fig.update_yaxes(title=axis_labels[1])

    return fig
