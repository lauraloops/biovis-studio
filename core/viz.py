import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import chi2

# ------------------------------------------------------------
# Confidence ellipse helper (ggplot2 stat_ellipse equivalent)
# ------------------------------------------------------------
def _confidence_ellipse(x, y, level=0.95):
    cov = np.cov(x, y)
    mean_x, mean_y = np.mean(x), np.mean(y)

    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]

    theta = np.linspace(0, 2 * np.pi, 200)
    circle = np.vstack((np.cos(theta), np.sin(theta)))

    scale = np.sqrt(chi2.ppf(level, df=2))
    ellipse = eigvecs @ np.diag(np.sqrt(eigvals)) @ circle * scale

    return ellipse[0] + mean_x, ellipse[1] + mean_y


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

    plot_df = df.copy()

    # --------------------------------------------------------
    # SAFE size handling (CRITICAL FIX)
    # --------------------------------------------------------
    if size is not None:
        s = plot_df[size].astype(float).values

        # Make values positive
        s = np.abs(s)

        # Handle constant / zero vectors
        if np.allclose(s, 0):
            s = np.ones_like(s)

        # Rescale to reasonable marker sizes
        s = 6 + 14 * (s - s.min()) / (s.max() - s.min())

        plot_df["_marker_size"] = s
        size_col = "_marker_size"
    else:
        size_col = None

    # --------------------------------------------------------
    # Scatter plot
    # --------------------------------------------------------
    fig = px.scatter(
        plot_df,
        x=x,
        y=y,
        color=color,
        symbol=symbol,
        size=size_col,
        hover_data=df.columns,
        opacity=0.85,
    )

    fig.update_traces(
        marker=dict(
            line=dict(width=0.6, color="black"),
            sizemode="area",
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
    # Confidence ellipses (group-wise)
    # --------------------------------------------------------
    if add_ellipses and color is not None:
        for group, gdf in plot_df.groupby(color):
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


def make_scatter(df: pd.DataFrame, x: str, y: str,
                 color_by: str | None = None,
                 shape_by: str | None = None,
                 size_by: str | None = None,
                 color_palette=None,
                 theme: str = "light",
                 title: str | None = None) -> go.Figure:
    """
    Minimal reusable scatter for AI Assistant and other pages.
    """
    color_arg = color_by if color_by else None
    symbol_arg = shape_by if shape_by else None
    size_arg = size_by if size_by else None

    fig = px.scatter(
        df,
        x=x, y=y,
        color=color_arg,
        symbol=symbol_arg,
        size=size_arg,
        color_discrete_sequence=color_palette if color_palette else px.colors.qualitative.Dark24,
        title=title
    )

    template = "plotly_white" if theme == "light" else "plotly_dark"
    plot_bg = "#ffffff" if theme == "light" else "#0e1117"
    paper_bg = plot_bg

    fig.update_layout(
        template=template,
        paper_bgcolor=paper_bg,
        plot_bgcolor=plot_bg,
        margin=dict(l=60, r=30, t=50, b=60),
        height=500
    )
    return fig