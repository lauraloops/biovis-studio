import os, sys
import pandas as pd
import streamlit as st

# ------------------------------------------------------------------
# Path fix so imports from core/ work both in Home.py and pages/*
# ------------------------------------------------------------------
HERE = os.path.dirname(__file__)
CANDIDATES = [
    os.path.abspath(os.path.join(HERE, "..")),
    os.path.abspath(os.path.join(HERE, "..", "..")),
]
for ROOT in CANDIDATES:
    if os.path.isdir(os.path.join(ROOT, "core")) and ROOT not in sys.path:
        sys.path.insert(0, ROOT)
        break

# ------------------------------------------------------------------
# Page config
# ------------------------------------------------------------------
st.set_page_config(page_title="BioVis Studio", layout="wide")

# ------------------------------------------------------------------
# Title & description
# ------------------------------------------------------------------
st.title("🧬 BioVis Studio")

st.write(
    "Upload a dataset → pick a preprocessing recipe → run PCA/UMAP → "
    "make plots → export a reproducible notebook."
)

st.markdown(
    "Use the left sidebar to navigate pages: "
    "**Import → Preprocess → Dimensionality Reduction → Visualize → Export**."
)

st.markdown("---")

# ------------------------------------------------------------------
# Session state initialization
# ------------------------------------------------------------------
if "df_raw" not in st.session_state:
    st.session_state["df_raw"] = None

if "df_processed" not in st.session_state:
    st.session_state["df_processed"] = None

if "preprocessing_steps" not in st.session_state:
    st.session_state["preprocessing_steps"] = []

# ------------------------------------------------------------------
# Data upload
# ------------------------------------------------------------------
st.subheader("📂 Upload data")

uploaded_file = st.file_uploader(
    "Upload a CSV, TSV, or Parquet file",
    type=["csv", "tsv", "parquet"],
)

if uploaded_file is not None:
    try:
        # Detect format
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
            file_type = "CSV"
        elif uploaded_file.name.endswith(".tsv"):
            df = pd.read_csv(uploaded_file, sep="\t")
            file_type = "TSV"
        else:
            df = pd.read_parquet(uploaded_file)
            file_type = "Parquet"

        # Basic validation
        if df.empty:
            st.error("The uploaded file is empty.")
        else:
            # Reset downstream state if new data is uploaded
            st.session_state["df_raw"] = df
            st.session_state["df_processed"] = None
            st.session_state["preprocessing_steps"] = []

            st.success(
                f"{file_type} loaded successfully: "
                f"{df.shape[0]} rows × {df.shape[1]} columns"
            )

            # Warnings for large or wide data
            if df.shape[1] > 10_000:
                st.warning(
                    "This dataset has a very large number of features. "
                    "Some operations may be slow."
                )

            # Preview
            st.markdown("**Preview (first 5 rows)**")
            st.dataframe(df.head(), use_container_width=True)

    except Exception as e:
        st.error(f"Failed to load file: {e}")

# ------------------------------------------------------------------
# Tips
# ------------------------------------------------------------------
st.markdown("---")
st.subheader("Tips")
st.markdown(
    "- Use tidy tables (samples × features).\n"
    "- Include sample-level metadata columns (e.g., Group, Batch).\n"
    "- For RNA-seq counts, try the RNA-seq preset in **Preprocess**."
)
