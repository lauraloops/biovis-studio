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

def fig_to_png_bytes(fig):
    # Requires kaleido
    return fig.to_image(format="png")
