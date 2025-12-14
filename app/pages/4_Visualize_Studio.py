import os, sys
HERE = os.path.dirname(__file__)
CANDIDATES = [
    os.path.abspath(os.path.join(HERE, "..")),
    os.path.abspath(os.path.join(HERE, "..", "..")),
]
for ROOT in CANDIDATES:
    if os.path.isdir(os.path.join(ROOT, "core")) and ROOT not in sys.path:
        sys.path.insert(0, ROOT)
        break

import json
import base64
from dataclasses import dataclass
from typing import Optional, List

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform

st.set_page_config(page_title="4) Visualize Studio", layout="wide")

@dataclass
class JournalStyle:
    name: str
    font: str
    font_size: int
    line_width: float
    marker_size: int
    grid: bool
    background: str
    paper_bg: str

JOURNAL_STYLES = {
    "Nature": JournalStyle("Nature", "Arial", 14, 2.0, 8, True, "rgba(255,255,255,0.05)", "#ffffff"),
    "Science": JournalStyle("Science", "Helvetica", 13, 1.8, 7, True, "rgba(255,255,255,0.05)", "#ffffff"),
    "IEEE": JournalStyle("IEEE", "Times New Roman", 12, 1.6, 7, True, "rgba(255,255,255,0.05)", "#ffffff"),
    "Plain": JournalStyle("Plain", "Arial", 12, 1.5, 6, True, "rgba(0,0,0,0)", "#ffffff"),
}

COLORBLIND_SAFE = {
    "Okabe-Ito": ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#000000"],
    "Viridis": px.colors.sequential.Viridis,
    "Cividis": px.colors.sequential.Cividis,
    "Plasma": px.colors.sequential.Plasma,
    "Turbo": px.colors.sequential.Turbo,
}

def format_sigfigs(x: float, sigfigs: int) -> str:
    if pd.isna(x) or not np.isfinite(x):
        return ""
    if x == 0:
        return "0"
    return f"{float(x):.{sigfigs}g}"

def make_ticks(vals, sigfigs):
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return [], []
    vmin, vmax = np.min(vals), np.max(vals)
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return [], []
    if np.isclose(vmin, vmax):
        eps = 1e-6 if vmax == 0 else abs(vmax) * 1e-6
        vmin, vmax = vmin - eps, vmax + eps
    tickvals = np.linspace(vmin, vmax, 6)
    ticktext = [format_sigfigs(v, sigfigs) for v in tickvals]
    return tickvals, ticktext

def perform_clustering(df: pd.DataFrame, method: str, **kwargs):
    """Perform clustering and return labels."""
    X = df.select_dtypes(include=[np.number]).values
    
    if method == "kmeans":
        n_clusters = kwargs.get("n_clusters", 3)
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = model.fit_predict(X)
        return labels, model
    
    elif method == "hierarchical":
        n_clusters = kwargs.get("n_clusters", 3)
        linkage_method = kwargs.get("linkage_method", "ward")
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method)
        labels = model.fit_predict(X)
        return labels, model
    
    elif method == "dbscan":
        eps = kwargs.get("eps", 0.5)
        min_samples = kwargs.get("min_samples", 5)
        model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = model.fit_predict(X)
        return labels, model
    
    return None, None

def create_dendrogram(df: pd.DataFrame, linkage_method: str = "ward"):
    """Create hierarchical clustering dendrogram."""
    X = df.select_dtypes(include=[np.number]).values
    Z = linkage(X, method=linkage_method)
    
    fig = go.Figure()
    
    # Calculate dendrogram
    dend = dendrogram(Z, no_plot=True)
    
    icoord = np.array(dend['icoord'])
    dcoord = np.array(dend['dcoord'])
    
    for i in range(len(icoord)):
        fig.add_trace(go.Scatter(
            x=icoord[i],
            y=dcoord[i],
            mode='lines',
            line=dict(color='rgb(100,100,100)', width=1),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    fig.update_layout(
        title="Hierarchical Clustering Dendrogram",
        xaxis=dict(title="Sample Index", showticklabels=False),
        yaxis=dict(title="Distance"),
        height=400
    )
    
    return fig

st.title("4️⃣ Visualize Studio")
st.caption("Advanced visualization lab with dimensionality reduction, clustering, and publication-ready export")

# Check for data from previous steps
if "prep_meta" not in st.session_state or st.session_state.get("prep_meta") is None:
    st.error("❌ No processed data found. Please complete steps 1-3 first!")
    
    with st.expander("🔍 Debug: Session State Contents"):
        st.write("Available keys:", list(st.session_state.keys()))
        if "X" in st.session_state:
            st.write("X shape:", st.session_state["X"].shape)
        if "metadata" in st.session_state:
            st.write("metadata shape:", st.session_state["metadata"].shape)
    
    st.info("💡 Go back to **Import and Profile** → **Data Preparation** → **DimReduction** to prepare your data.")
    st.stop()

# Load data from session state
df = st.session_state["prep_meta"].copy()
metadata_cols = st.session_state.get("metadata_cols", [])

st.success(f"✅ Loaded {len(df)} samples with {len(df.columns)} features from previous steps")

# Sidebar configuration
with st.sidebar:
    st.subheader("📊 Visualization Settings")
    
    # Theme
    theme_bg = st.selectbox("Background theme", ["light", "dark"], index=0)
    
    # Determine available coordinate columns
    coord_options = []
    if "PC1" in df.columns and "PC2" in df.columns:
        coord_options.extend([("PC1", "PC2", "PCA")])
    if "UMAP1" in df.columns and "UMAP2" in df.columns:
        coord_options.extend([("UMAP1", "UMAP2", "UMAP")])
    if "TSNE1" in df.columns and "TSNE2" in df.columns:
        coord_options.extend([("TSNE1", "TSNE2", "t-SNE")])
    
    if not coord_options:
        st.error("No dimensionality reduction coordinates found. Run PCA/UMAP/t-SNE first.")
        st.stop()
    
    coord_choice = st.selectbox(
        "Coordinate system",
        options=range(len(coord_options)),
        format_func=lambda i: coord_options[i][2]
    )
    
    x_col, y_col, coord_label = coord_options[coord_choice]
    
    # Color/Shape/Size mappings
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in metadata_cols if c in df.columns]
    
    color_by = st.selectbox("Color by", options=["None"] + categorical_cols + numeric_cols, index=0)
    shape_by = st.selectbox("Shape by", options=["None"] + categorical_cols, index=0)
    size_by = st.selectbox("Size by", options=["None"] + numeric_cols, index=0)
    
    st.divider()
    
    # Style presets
    st.subheader("🎨 Style")
    style_name = st.selectbox("Journal preset", list(JOURNAL_STYLES.keys()), index=3)
    palette_name = st.selectbox("Color palette", list(COLORBLIND_SAFE.keys()), index=0)
    
    st.divider()
    
    # Formatting
    st.subheader("📐 Formatting")
    sigfigs = st.slider("Significant figures", min_value=2, max_value=6, value=3)
    x_label = st.text_input("X label", value=x_col)
    y_label = st.text_input("Y label", value=y_col)
    show_grid = st.checkbox("Show grid", value=True)
    show_legend = st.checkbox("Show legend", value=True)

# Main content tabs
tab_scatter, tab_clustering, tab_stats, tab_export = st.tabs([
    "📊 Scatter Plot",
    "🔬 Clustering Analysis",
    "📈 Statistics & Overlays",
    "💾 Export"
])

# ==================== TAB 1: SCATTER PLOT ====================
with tab_scatter:
    st.subheader(f"{coord_label} Visualization")
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        st.markdown("### Plot Options")
        marker_size = st.slider("Marker size", 3, 20, 8)
        marker_opacity = st.slider("Opacity", 0.1, 1.0, 0.85, 0.05)
        marker_outline = st.checkbox("Marker outline", value=False)
    
    with col1:
        # Build figure
        style = JOURNAL_STYLES[style_name]
        palette = COLORBLIND_SAFE[palette_name]
        
        color_arg = None if color_by == "None" else color_by
        symbol_arg = None if shape_by == "None" else shape_by
        size_arg = None if size_by == "None" else size_by
        
        fig = px.scatter(
            df,
            x=x_col,
            y=y_col,
            color=color_arg,
            symbol=symbol_arg,
            size=size_arg,
            color_discrete_sequence=palette if color_arg in categorical_cols or color_arg is None else None,
            color_continuous_scale="Viridis" if color_arg in numeric_cols else None,
            hover_data={c: True for c in metadata_cols if c in df.columns}
        )
        
        # Apply theme
        template = "plotly_white" if theme_bg == "light" else "plotly_dark"
        plot_bg = "#ffffff" if theme_bg == "light" else "#0e1117"
        paper_bg = "#ffffff" if theme_bg == "light" else "#0e1117"
        grid_color = "rgba(0,0,0,0.1)" if theme_bg == "light" else "rgba(255,255,255,0.1)"
        
        fig.update_layout(
            template=template,
            font=dict(family=style.font, size=style.font_size),
            paper_bgcolor=paper_bg,
            plot_bgcolor=plot_bg,
            legend=dict(bgcolor='rgba(0,0,0,0)') if show_legend else dict(showlegend=False),
            margin=dict(l=60, r=30, t=50, b=60),
            height=600
        )
        
        fig.update_traces(
            marker=dict(
                size=marker_size,
                opacity=marker_opacity,
                line=dict(width=1, color='white' if theme_bg == "dark" else 'black') if marker_outline else dict(width=0)
            )
        )
        
        fig.update_xaxes(
            title=x_label,
            showgrid=show_grid,
            gridcolor=grid_color,
            zeroline=False
        )
        
        fig.update_yaxes(
            title=y_label,
            showgrid=show_grid,
            gridcolor=grid_color,
            zeroline=False
        )
        
        # Format ticks with significant figures
        x_tickvals, x_ticktext = make_ticks(df[x_col].values, sigfigs)
        y_tickvals, y_ticktext = make_ticks(df[y_col].values, sigfigs)
        fig.update_xaxes(tickvals=x_tickvals, ticktext=x_ticktext)
        fig.update_yaxes(tickvals=y_tickvals, ticktext=y_ticktext)
        
        st.plotly_chart(fig, use_container_width=True, key="main_scatter")
        
        # Store figure in session for export
        st.session_state["viz_figure"] = fig

# ==================== TAB 2: CLUSTERING ====================
with tab_clustering:
    st.subheader("🔬 Clustering Analysis")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Clustering Method")
        cluster_method = st.selectbox(
            "Algorithm",
            ["kmeans", "hierarchical", "dbscan"],
            format_func=lambda x: {
                "kmeans": "K-Means",
                "hierarchical": "Hierarchical (Agglomerative)",
                "dbscan": "DBSCAN (Density-based)"
            }[x]
        )
        
        if cluster_method == "kmeans":
            n_clusters = st.slider("Number of clusters", 2, 10, 3)
            kwargs = {"n_clusters": n_clusters}
        
        elif cluster_method == "hierarchical":
            n_clusters = st.slider("Number of clusters", 2, 10, 3)
            linkage_method = st.selectbox("Linkage", ["ward", "complete", "average", "single"])
            kwargs = {"n_clusters": n_clusters, "linkage_method": linkage_method}
        
        elif cluster_method == "dbscan":
            eps = st.slider("Epsilon (neighborhood radius)", 0.1, 5.0, 0.5, 0.1)
            min_samples = st.slider("Min samples", 2, 20, 5)
            kwargs = {"eps": eps, "min_samples": min_samples}
        
        run_clustering = st.button("🔬 Run Clustering", type="primary")
    
    with col2:
        if run_clustering:
            with st.spinner("Running clustering..."):
                # Use coordinates for clustering
                cluster_df = df[[x_col, y_col]].copy()
                labels, model = perform_clustering(cluster_df, cluster_method, **kwargs)
                
                if labels is not None:
                    df["cluster"] = labels.astype(str)
                    st.session_state["cluster_labels"] = labels
                    
                    # Show cluster distribution
                    unique_labels = np.unique(labels)
                    n_clusters_found = len(unique_labels[unique_labels >= 0])  # Exclude noise (-1 for DBSCAN)
                    
                    st.success(f"✅ Found {n_clusters_found} clusters")
                    
                    # Cluster size distribution
                    cluster_counts = pd.Series(labels).value_counts().sort_index()
                    st.dataframe(
                        pd.DataFrame({
                            "Cluster": cluster_counts.index,
                            "Size": cluster_counts.values,
                            "%": (cluster_counts.values / len(labels) * 100).round(1)
                        }),
                        hide_index=True
                    )
                    
                    # Visualize clusters
                    fig_cluster = px.scatter(
                        df,
                        x=x_col,
                        y=y_col,
                        color="cluster",
                        title=f"{cluster_method.upper()} Clustering Results",
                        color_discrete_sequence=COLORBLIND_SAFE[palette_name]
                    )
                    
                    template = "plotly_white" if theme_bg == "light" else "plotly_dark"
                    plot_bg = "#ffffff" if theme_bg == "light" else "#0e1117"
                    paper_bg = "#ffffff" if theme_bg == "light" else "#0e1117"
                    
                    fig_cluster.update_layout(
                        template=template,
                        paper_bgcolor=paper_bg,
                        plot_bgcolor=plot_bg,
                        height=500
                    )
                    
                    st.plotly_chart(fig_cluster, use_container_width=True)
                    
                    # Dendrogram for hierarchical
                    if cluster_method == "hierarchical":
                        st.subheader("Dendrogram")
                        fig_dend = create_dendrogram(cluster_df, kwargs.get("linkage_method", "ward"))
                        fig_dend.update_layout(
                            template=template,
                            paper_bgcolor=paper_bg,
                            plot_bgcolor=plot_bg
                        )
                        st.plotly_chart(fig_dend, use_container_width=True)

# ==================== TAB 3: STATISTICS ====================
with tab_stats:
    st.subheader("📈 Statistical Overlays")
    
    col1, col2 = st.columns(2)
    
    with col1:
        add_regression = st.checkbox("Add OLS regression line", value=False)
        if add_regression:
            show_ci = st.checkbox("Show 95% confidence band", value=True)
    
    with col2:
        add_density = st.checkbox("Add density contours", value=False)
        add_ellipse = st.checkbox("Add confidence ellipses (by group)", value=False)
    
    if add_regression or add_density or add_ellipse:
        fig_stats = go.Figure(st.session_state.get("viz_figure", go.Figure()))
        
        # Regression
        if add_regression and len(df) > 2:
            x = df[x_col].values
            y = df[y_col].values
            mask = np.isfinite(x) & np.isfinite(y)
            x, y = x[mask], y[mask]
            
            A = np.vstack([x, np.ones_like(x)]).T
            beta, resid, _, _ = np.linalg.lstsq(A, y, rcond=None)
            slope, intercept = beta
            x_line = np.linspace(np.nanmin(x), np.nanmax(x), 200)
            y_line = slope * x_line + intercept
            
            line_color = "black" if theme_bg == "light" else "white"
            
            fig_stats.add_trace(go.Scatter(
                x=x_line, y=y_line,
                mode="lines",
                line=dict(color=line_color, width=2, dash="dash"),
                name="OLS fit"
            ))
            
            if show_ci and resid.size > 0:
                n = len(x)
                sigma2 = resid[0] / max(n - 2, 1)
                x_mean = np.mean(x)
                Sxx = np.sum((x - x_mean) ** 2)
                se_pred = np.sqrt(sigma2 * (1 + (1/n) + (x_line - x_mean) ** 2 / Sxx))
                ci = 1.96 * se_pred
                
                fig_stats.add_trace(go.Scatter(
                    x=np.concatenate([x_line, x_line[::-1]]),
                    y=np.concatenate([y_line + ci, (y_line - ci)[::-1]]),
                    fill='toself',
                    fillcolor='rgba(100,100,100,0.2)' if theme_bg == "light" else 'rgba(200,200,200,0.15)',
                    line=dict(color='rgba(0,0,0,0)'),
                    name="95% CI"
                ))
        
        # Density contours
        if add_density:
            try:
                fig_density = px.density_contour(df, x=x_col, y=y_col)
                for trace in fig_density.data:
                    fig_stats.add_trace(trace)
            except Exception as e:
                st.warning(f"Could not add density contours: {e}")
        
        template = "plotly_white" if theme_bg == "light" else "plotly_dark"
        plot_bg = "#ffffff" if theme_bg == "light" else "#0e1117"
        paper_bg = "#ffffff" if theme_bg == "light" else "#0e1117"
        
        fig_stats.update_layout(
            template=template,
            paper_bgcolor=paper_bg,
            plot_bgcolor=plot_bg,
            height=600
        )
        
        st.plotly_chart(fig_stats, use_container_width=True)

# ==================== TAB 4: EXPORT ====================
with tab_export:
    st.subheader("💾 Export Publication-Ready Figure")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        dpi = st.number_input("DPI (PNG)", min_value=100, max_value=600, value=300, step=50)
    
    with col2:
        fmt = st.selectbox("Format", ["PNG", "SVG", "PDF"], index=0)
    
    with col3:
        include_spec = st.checkbox("Include JSON spec", value=True)
    
    # Spec for reproducibility
    spec = {
        "coordinate_system": coord_label,
        "columns": {"x": x_col, "y": y_col, "color": color_by, "shape": shape_by, "size": size_by},
        "formatting": {"sigfigs": sigfigs, "x_label": x_label, "y_label": y_label},
        "style": {"journal": style_name, "palette": palette_name, "theme": theme_bg, "grid": show_grid},
        "data_source": "session_state.prep_meta",
        "n_samples": len(df)
    }
    
    if st.button("📥 Export Figure", type="primary", use_container_width=True):
        try:
            fig_to_export = st.session_state.get("viz_figure")
            
            if fig_to_export is None:
                st.error("No figure to export. Generate a plot first.")
            else:
                if fmt == "PNG":
                    img_bytes = fig_to_export.to_image(format="png", scale=max(dpi/96, 1.0))
                    mime = "image/png"
                elif fmt == "SVG":
                    img_bytes = fig_to_export.to_image(format="svg")
                    mime = "image/svg+xml"
                elif fmt == "PDF":
                    img_bytes = fig_to_export.to_image(format="pdf")
                    mime = "application/pdf"
                
                fname = f"figure_{coord_label}_{x_col}_vs_{y_col}.{fmt.lower()}"
                
                st.download_button(
                    label=f"⬇️ Download {fname}",
                    data=img_bytes,
                    file_name=fname,
                    mime=mime
                )
                
                if include_spec:
                    spec_str = json.dumps(spec, indent=2)
                    st.download_button(
                        label="⬇️ Download figure.spec.json",
                        data=spec_str.encode(),
                        file_name="figure.spec.json",
                        mime="application/json"
                    )
                
                st.success("✅ Export ready!")
        
        except Exception as e:
            st.error(f"❌ Export failed: {e}")
            st.exception(e)
    
    # Export data
    st.divider()
    st.subheader("📊 Export Data")
    
    if st.button("💾 Download processed data (CSV)"):
        csv = df.to_csv(index=True).encode('utf-8')
        st.download_button(
            label="⬇️ Download CSV",
            data=csv,
            file_name=f"processed_data_{coord_label}.csv",
            mime="text/csv"
        )
