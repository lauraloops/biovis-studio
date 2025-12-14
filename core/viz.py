import pandas as pd
import plotly.express as px

def make_scatter(meta: pd.DataFrame, color_by=None, shape_by=None, size_by=None):
    # Expect columns PC1/PC2 or UMAP/t-SNE in meta; fallback to PC1/PC2
    x_col = None
    y_col = None
    for pair in [("UMAP1","UMAP2"),("TSNE1","TSNE2"),("PC1","PC2")]:
        if all(p in meta.columns for p in pair):
            x_col, y_col = pair
            break
    if x_col is None:
        # create a trivial index-based scatter
        meta = meta.copy()
        meta["x"], meta["y"] = range(len(meta)), range(len(meta))
        x_col, y_col = "x", "y"
    fig = px.scatter(meta, x=x_col, y=y_col, color=color_by, symbol=shape_by, size=size_by)
    fig.update_layout(margin=dict(l=10,r=10,t=40,b=10))
    return fig

def plot_embedding(
    df,
    x,
    y,
    color=None,
    symbol=None,
    title=None,
):
    fig = px.scatter(
        df,
        x=x,
        y=y,
        color=color,
        symbol=symbol,
        hover_data=df.columns,
        opacity=0.85,
    )

    fig.update_traces(
        marker=dict(
            size=8,
            line=dict(width=0.5, color="black")
        )
    )

    fig.update_layout(
        title=title,
        template="simple_white",
        legend_title_text="",
        font=dict(size=14),
        margin=dict(l=10, r=10, t=50, b=10),
    )

    return fig