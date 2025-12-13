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

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.impute import SimpleImputer

from core.data_quality import (
    detect_outliers_univariate, detect_outliers_multivariate,
    summarize_outliers, identify_missing_data_patterns, calculate_qc_metrics
)
from core.normalization import normalize_data, get_normalization_recommendations, detect_batch_effects

st.set_page_config(page_title="Data Preparation", layout="wide")
st.header("2️⃣ Data Preparation & Quality Control")

if "working_df" not in st.session_state:
    st.error("❌ Please upload and profile data in step 1 first!")
    st.stop()

working_df = st.session_state["working_df"]
metadata_cols = st.session_state.get("metadata_cols", [])
feature_cols = st.session_state.get("feature_cols", 
    [c for c in working_df.columns if pd.api.types.is_numeric_dtype(working_df[c])])

numeric_df = working_df[feature_cols].select_dtypes(include=[np.number])

st.sidebar.markdown("### 📋 Data Summary")
st.sidebar.metric("Samples (rows)", numeric_df.shape[0])
st.sidebar.metric("Features (cols)", numeric_df.shape[1])

# ==================== TABS ====================
tab_missing, tab_qc, tab_outliers, tab_normalization, tab_review = st.tabs(
    ["🔍 Missing Data", "📈 QC Metrics", "⚠️ Outliers", "📊 Normalization", "✅ Review & Apply"]
)

# ==================== TAB 1: MISSING DATA ====================
with tab_missing:
    st.subheader("🔍 Missing Data Analysis")
    
    missing_info = identify_missing_data_patterns(numeric_df)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Missing Values", missing_info["total_missing"])
    with col2:
        st.metric("Missing %", f"{missing_info['total_missing_%']}%")
    with col3:
        st.metric("Rows with Missing Data", missing_info["rows_with_any_missing"])
    
    if missing_info["columns_with_missing"]:
        st.subheader("Missing Data by Feature")
        missing_df = pd.DataFrame([
            {"feature": k, "count": v, "%": round(v/len(numeric_df)*100, 2)}
            for k, v in missing_info["columns_with_missing"].items()
        ]).sort_values("count", ascending=False)
        
        st.dataframe(missing_df.head(20), use_container_width=True, hide_index=True)
        
        fig = px.bar(
            missing_df.head(20),
            x="feature", y="count",
            title="Top 20 Features with Missing Values",
            labels={"count": "# Missing Values"},
            color="count",
            color_continuous_scale="Reds"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.success("✅ No missing values detected!")
    
    st.subheader("Missing Data Handling Strategy")
    missing_strategy = st.radio(
        "How to handle missing values?",
        [
            "Keep all (impute during normalization)",
            "Remove rows with ANY missing",
            "Remove features with >50% missing"
        ],
        horizontal=True
    )
    
    if st.button("🔧 Apply Missing Data Strategy"):
        if missing_strategy == "Keep all (impute during normalization)":
            st.session_state["missing_strategy"] = "keep"
            st.info("✅ Keeping all data. Imputation will happen during normalization.")
        elif missing_strategy == "Remove rows with ANY missing":
            before = len(numeric_df)
            clean_idx = numeric_df.dropna().index
            working_df = working_df.loc[clean_idx]
            st.session_state["working_df"] = working_df
            after = len(clean_idx)
            st.success(f"✅ Removed {before - after} rows ({round((before-after)/before*100, 1)}%)")
            st.session_state["missing_strategy"] = "rows_removed"
            st.rerun()
        elif missing_strategy == "Remove features with >50% missing":
            high_missing = [k for k, v in missing_info["columns_with_missing"].items() 
                          if v/len(numeric_df) > 0.5]
            if high_missing:
                working_df = working_df.drop(columns=high_missing)
                st.session_state["working_df"] = working_df
                st.session_state["feature_cols"] = [c for c in working_df.columns if c not in metadata_cols]
                st.success(f"✅ Removed {len(high_missing)} features: {', '.join(high_missing)}")
            else:
                st.info("✅ No features with >50% missing data.")
            st.session_state["missing_strategy"] = "cols_removed"
            st.rerun()

# ==================== TAB 2: QC METRICS ====================
with tab_qc:
    st.subheader("📈 Quality Control Metrics")
    
    qc_results = calculate_qc_metrics(numeric_df)
    sample_qc = qc_results["sample_qc"]
    feature_qc = qc_results["feature_qc"]
    lib_stats = qc_results["library_size_stats"]
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Min Library Size", f"{int(lib_stats['min']):,}")
    with col2:
        st.metric("Max Library Size", f"{int(lib_stats['max']):,}")
    with col3:
        st.metric("Median Library Size", f"{int(lib_stats['median']):,}")
    with col4:
        st.metric("Mean Library Size", f"{int(lib_stats['mean']):,}")
    
    st.subheader("Sample QC Metrics")
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.scatter(
            sample_qc, x="library_size", y="detection_rate_%",
            title="Library Size vs Detection Rate",
            labels={"library_size": "Library Size", "detection_rate_%": "Detection Rate (%)"},
            color="n_detected_features",
            color_continuous_scale="Viridis"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.histogram(
            sample_qc, x="detection_rate_%",
            title="Distribution of Detection Rates",
            nbins=30,
            labels={"detection_rate_%": "Detection Rate (%)"}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Feature QC Metrics")
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.scatter(
            feature_qc.head(100), x="mean_expression", y="cv",
            title="Mean Expression vs Coefficient of Variation (Top 100)",
            labels={"mean_expression": "Mean Expression", "cv": "CV"},
            size="detection_rate_%",
            color="detection_rate_%"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.histogram(
            feature_qc, x="detection_rate_%",
            title="Distribution of Feature Detection Rates",
            nbins=30,
            labels={"detection_rate_%": "Detection Rate (%)"}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Low quality features
    st.subheader("Low Quality Features (Manual Review)")
    low_quality = feature_qc[feature_qc["detection_rate_%"] < 10].sort_values("detection_rate_%")
    if len(low_quality) > 0:
        st.warning(f"⚠️ {len(low_quality)} features detected in <10% of samples")
        st.dataframe(low_quality.head(20), use_container_width=True, hide_index=True)
    else:
        st.success("✅ All features detected in >10% of samples")
    
    # Batch effect detection
    if len(metadata_cols) > 0:
        st.subheader("Batch Effect Detection")
        batch_col = st.selectbox("Select batch/condition column:", metadata_cols)
        
        if batch_col:
            batch_stats = detect_batch_effects(numeric_df, working_df[batch_col])
            batch_df = pd.DataFrame(batch_stats).T
            st.dataframe(batch_df, use_container_width=True)

# ==================== TAB 3: OUTLIERS ====================
with tab_outliers:
    st.subheader("⚠️ Outlier Detection & Visualization")
    
    col_method, col_param = st.columns(2)
    
    with col_method:
        univariate_method = st.selectbox(
            "Univariate Method (per-feature):",
            ["iqr", "zscore", "mad"],
            help="""
            **IQR:** Interquartile range. Flags points beyond Q1-k*IQR and Q3+k*IQR.
            **Z-score:** Flags points |z| > threshold std devs from mean.
            **MAD:** Median Absolute Deviation. Most robust to extreme outliers.
            """
        )
    
    with col_param:
        if univariate_method == "iqr":
            iqr_k = st.slider("IQR multiplier (k)", 1.0, 3.0, 1.5, 0.1)
            outliers_uni = detect_outliers_univariate(numeric_df, method="iqr", iqr_k=iqr_k)
        elif univariate_method == "zscore":
            zscore_threshold = st.slider("Z-score threshold", 1.0, 5.0, 3.0, 0.5)
            outliers_uni = detect_outliers_univariate(numeric_df, method="zscore", zscore_threshold=zscore_threshold)
        else:  # mad
            mad_threshold = st.slider("MAD threshold", 1.0, 5.0, 3.5, 0.2)
            outliers_uni = detect_outliers_univariate(numeric_df, method="mad", mad_threshold=mad_threshold)
    
    outlier_summary = summarize_outliers(numeric_df, outliers_uni)
    
    st.subheader(f"Univariate Outliers ({univariate_method.upper()})")
    
    col_table, col_chart = st.columns([1, 2])
    
    with col_table:
        st.dataframe(outlier_summary.head(15), use_container_width=True, hide_index=True)
    
    with col_chart:
        if len(outlier_summary) > 0:
            fig = px.bar(
                outlier_summary.head(10),
                x="feature", y="outliers_count",
                title="Top 10 Features with Outliers",
                color="outliers_%",
                color_continuous_scale="YlOrRd"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Boxplots for top features
    if len(outlier_summary) > 0:
        st.subheader("Boxplot Visualization (Top 6 Features)")
        top_features = outlier_summary.head(6)["feature"].tolist()
        
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=top_features,
            specs=[[{"type": "box"}]*3]*2
        )
        
        for idx, feat in enumerate(top_features):
            row = idx // 3 + 1
            col = idx % 3 + 1
            
            fig.add_trace(
                go.Box(y=numeric_df[feat].dropna(), name=feat, marker_color="lightblue"),
                row=row, col=col
            )
        
        fig.update_layout(height=600, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # Multivariate
    st.subheader("⚠️ Multivariate Outlier Detection")
    
    multivar_method = st.radio(
        "Multivariate method:",
        ["Isolation Forest", "PCA-based"],
        horizontal=True
    )
    
    if multivar_method == "Isolation Forest":
        contamination = st.slider(
            "Expected contamination %",
            0.1, 20.0, 5.0, 0.5
        )
        
        if st.button("🔎 Run Multivariate Detection (Isolation Forest)"):
            outliers_multi = detect_outliers_multivariate(numeric_df, method="isolation_forest", contamination=contamination/100)
            st.session_state["outliers_multi"] = outliers_multi
            outlier_count = outliers_multi.sum()
            st.success(f"✅ Found {outlier_count} multivariate outliers ({round(outlier_count/len(numeric_df)*100, 1)}%)")
    
    else:  # PCA-based
        n_comp = st.slider("PCA components", 2, min(10, numeric_df.shape[1]), 2)
        pca_thresh = st.slider("Reconstruction error threshold (σ)", 1.0, 5.0, 2.0, 0.5)
        
        if st.button("🔎 Run Multivariate Detection (PCA)"):
            outliers_multi = detect_outliers_multivariate(numeric_df, method="pca", n_components=n_comp, pca_threshold=pca_thresh)
            st.session_state["outliers_multi"] = outliers_multi
            outlier_count = outliers_multi.sum()
            st.success(f"✅ Found {outlier_count} PCA-based outliers ({round(outlier_count/len(numeric_df)*100, 1)}%)")
    
    # Decision
    st.subheader("📋 Outlier Handling Decision")
    outlier_action = st.radio(
        "What to do with flagged outliers?",
        [
            "Keep all (for sensitivity analysis)",
            "Remove flagged rows",
            "Flag for visualization (keep in data)"
        ],
        horizontal=True
    )
    
    st.session_state["outlier_action"] = outlier_action
    st.session_state["outliers_univariate"] = outliers_uni

# ==================== TAB 4: NORMALIZATION ====================
with tab_normalization:
    st.subheader("📊 Normalization & Transformation")
    
    # Get recommendations
    recommendations = get_normalization_recommendations(numeric_df)
    
    col1, col2 = st.columns(2)
    
    with col1:
        norm_method = st.selectbox(
            "Normalization Method:",
            ["standard", "robust", "minmax", "log1p", "log2p", "yeo-johnson", "box-cox", "none"],
            help="""
            **standard:** (x - mean) / std. Best for normally distributed data.
            **robust:** Uses median & IQR. Better with outliers.
            **minmax:** Scales to [0, 1]. Good for bounded interpretability.
            **log1p:** log(1 + x). For count/RNA-seq data.
            **log2p:** log₂(1 + x). Common in genomics.
            **yeo-johnson:** Power transform. Handles negatives.
            **box-cox:** Power transform. Positive data only.
            **none:** No transformation.
            """
        )
    
    with col2:
        impute_method = st.selectbox(
            "Missing Value Imputation:",
            ["median", "mean", "none"],
            help="Applied before normalization"
        )
    
    st.subheader("📈 Method Recommendations")
    if recommendations:
        rec_df = pd.DataFrame([
            {"feature": k, "recommended": v}
            for k, v in list(recommendations.items())[:10]
        ])
        st.dataframe(rec_df, use_container_width=True, hide_index=True)
    
    # Apply & preview
    if st.button("📊 Preview Normalization"):
        # Impute if needed
        if impute_method != "none":
            imputer = SimpleImputer(strategy=impute_method)
            numeric_df_imputed = pd.DataFrame(
                imputer.fit_transform(numeric_df),
                columns=numeric_df.columns,
                index=numeric_df.index
            )
        else:
            numeric_df_imputed = numeric_df.fillna(numeric_df.mean())
        
        normalized, metadata = normalize_data(numeric_df_imputed, method=norm_method)
        st.session_state["normalized_df"] = normalized
        st.session_state["normalization_method"] = norm_method
        st.session_state["impute_method"] = impute_method
        st.session_state["norm_metadata"] = metadata
        
        # Show comparison for first numeric column
        cols_to_compare = list(numeric_df.columns)[:3]
        
        if len(cols_to_compare) > 0:
            sample_col = cols_to_compare[0]
            
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=("Before Normalization", "After Normalization"),
                specs=[[{"type": "histogram"}, {"type": "histogram"}]]
            )
            
            fig.add_trace(
                go.Histogram(x=numeric_df[sample_col].dropna(), name="Original", nbinsx=30, marker_color="steelblue"),
                row=1, col=1
            )
            fig.add_trace(
                go.Histogram(x=normalized[sample_col], name="Normalized", nbinsx=30, marker_color="orange"),
                row=1, col=2
            )
            
            fig.update_xaxes(title_text="Value", row=1, col=1)
            fig.update_xaxes(title_text="Normalized Value", row=1, col=2)
            fig.update_layout(height=400)
            
            st.plotly_chart(fig, use_container_width=True)
        
        st.success("✅ Normalization preview ready!")

# ==================== TAB 5: REVIEW & APPLY ====================
with tab_review:
    st.subheader("✅ Review & Apply All Changes")
    
    st.write("**Summary of operations:**")
    
    config_col1, config_col2 = st.columns(2)
    
    with config_col1:
        st.markdown("### Missing Data")
        st.write(f"**Strategy:** {st.session_state.get('missing_strategy', '❌ Not set')}")
        
        st.markdown("### Outliers")
        st.write(f"**Action:** {st.session_state.get('outlier_action', '❌ Not set')}")
    
    with config_col2:
        st.markdown("### Normalization")
        st.write(f"**Method:** {st.session_state.get('normalization_method', '❌ Not set')}")
        
        st.markdown("### Imputation")
        st.write(f"**Method:** {st.session_state.get('impute_method', '❌ Not set')}")
    
    st.divider()
    
    if st.button("✅ APPLY ALL & PROCEED", type="primary", use_container_width=True):
        # Get final working dataframe
        df_final = st.session_state["working_df"].copy()
        
        # Get feature columns (numeric only)
        final_metadata_cols = st.session_state.get("metadata_cols", [])
        final_feature_cols = [c for c in df_final.columns if c not in final_metadata_cols]
        numeric_df_final = df_final[final_feature_cols].select_dtypes(include=[np.number])
        
        # Imputation
        impute = st.session_state.get("impute_method", "median")
        if impute != "none":
            imputer = SimpleImputer(strategy=impute)
            numeric_df_final = pd.DataFrame(
                imputer.fit_transform(numeric_df_final),
                columns=numeric_df_final.columns,
                index=numeric_df_final.index
            )
        
        # Normalization
        norm_method = st.session_state.get("normalization_method", "standard")
        normalized_df, norm_meta = normalize_data(numeric_df_final, method=norm_method)
        
        # Handle outliers
        outlier_action = st.session_state.get("outlier_action", "Keep all")
        outlier_flag = None
        
        if "Remove flagged rows" in outlier_action:
            outliers = st.session_state.get("outliers_univariate", {})
            any_outlier = np.zeros(len(normalized_df), dtype=bool)
            for flags in outliers.values():
                any_outlier |= flags
            normalized_df = normalized_df[~any_outlier]
            st.session_state["rows_removed_outliers"] = any_outlier.sum()
        elif "Flag for visualization" in outlier_action:
            outliers = st.session_state.get("outliers_univariate", {})
            any_outlier = np.zeros(len(normalized_df), dtype=bool)
            for flags in outliers.values():
                any_outlier |= flags
            outlier_flag = any_outlier
        
        # Get metadata for remaining samples
        meta = df_final.loc[normalized_df.index][final_metadata_cols].copy() if final_metadata_cols else pd.DataFrame(index=normalized_df.index)
        
        # Store results
        st.session_state["X"] = normalized_df
        st.session_state["metadata"] = meta
        st.session_state["outlier_flag"] = outlier_flag
        st.session_state["prep_complete"] = True
        
        st.success(f"""
        ✅ **Data Preparation Complete!**
        
        - **Final samples:** {normalized_df.shape[0]}
        - **Final features:** {normalized_df.shape[1]}
        - **Normalization:** {norm_method}
        - **Imputation:** {impute}
        """)
        
        st.info("🚀 Ready for **Dimensionality Reduction** (PCA, UMAP, t-SNE)")