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

import pandas as pd
import streamlit as st
from core.io import read_any, infer_schema, detect_id_columns, detect_duplicate_ids, infer_orientation

st.set_page_config(page_title="Import & Profile", layout="wide")
st.header("1️⃣ Import & Profile Data")

uploaded = st.file_uploader(
    "📤 Upload CSV / TSV / Parquet",
    type=["csv", "tsv", "txt", "parquet"]
)

if uploaded:
    df = read_any(uploaded)
    st.session_state["raw_df"] = df
    st.session_state["working_df"] = df.copy()
    
    tab_profile, tab_structure, tab_metadata = st.tabs(
        ["📊 Profile", "🔧 Structure & Orientation", "🏷️ Metadata Columns"]
    )
    
    with tab_profile:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📝 Rows", df.shape[0])
        with col2:
            st.metric("📋 Columns", df.shape[1])
        with col3:
            st.metric("📦 Size (MB)", round(df.memory_usage(deep=True).sum() / 1e6, 2))
        
        st.subheader("Data Schema")
        schema = infer_schema(df)
        st.dataframe(schema, use_container_width=True, hide_index=True)
        
        st.subheader("Preview (First 20 rows)")
        st.dataframe(df.head(20), use_container_width=True)
    
    with tab_structure:
        st.subheader("Data Orientation Analysis")
        orientation = infer_orientation(df)
        st.info(f"**Detected:** {orientation['reason']}")
        st.write(f"**Recommendation:** {orientation['recommendation']}")
        
        if orientation["orientation"] == "features_x_samples":
            if st.button("🔄 Transpose Data"):
                df = df.T
                st.session_state["raw_df"] = df
                st.session_state["working_df"] = df.copy()
                st.success("✅ Data transposed!")
                st.rerun()
        
        # Detect duplicate columns
        st.subheader("Duplicate Column Detection")
        duplicates = detect_duplicate_ids(df)
        if duplicates:
            st.warning(f"⚠️ Detected duplicate/highly-correlated columns:")
            for pair, type_ in duplicates.items():
                st.write(f"- {pair} ({type_})")
        else:
            st.success("✅ No duplicate columns detected.")
    
    with tab_metadata:
        st.subheader("🏷️ Identify ID & Metadata Columns")
        
        id_candidates = detect_id_columns(df, threshold=0.95)
        if id_candidates:
            st.warning(f"🔍 **Detected likely ID columns:** {', '.join(id_candidates)}")
        
        st.subheader("Select columns to keep")
        all_cols = list(df.columns)
        cols_to_keep = st.multiselect(
            "Keep these columns:",
            all_cols,
            default=all_cols,
            help="Uncheck ID/index columns to remove them"
        )
        
        if len(cols_to_keep) < len(all_cols):
            if st.button("✂️ Remove unchecked columns"):
                cols_to_remove = [c for c in all_cols if c not in cols_to_keep]
                df = df[cols_to_keep]
                st.session_state["working_df"] = df
                st.session_state["raw_df"] = df
                st.success(f"✅ Removed: {cols_to_remove}")
                st.rerun()
        
        st.subheader("🏷️ Classify Metadata Columns")
        st.caption("These columns won't be analyzed but will be preserved for visualization")
        
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        default_metadata = [c for c in df.columns if c not in numeric_cols]
        
        selected_metadata = st.multiselect(
            "Select metadata columns (group, batch, condition, cell_type, etc.):",
            df.columns,
            default=default_metadata if default_metadata else []
        )
        
        st.session_state["metadata_cols"] = selected_metadata
        st.session_state["feature_cols"] = [c for c in df.columns if c not in selected_metadata]
        
        if selected_metadata:
            st.dataframe(df[selected_metadata].head(10), use_container_width=True)
    
    st.info("✅ Next: Go to **Data Preparation** to handle missing data, outliers, and normalization.")
else:
    st.info("""
    📁 Upload a CSV, TSV, or Parquet file to begin.
    
    **Sample data in this project:**
    - `/data/Heart.csv`
    - `/data/demo_expression.csv`
    """)
