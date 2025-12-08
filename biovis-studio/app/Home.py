import os, sys
HERE = os.path.dirname(__file__)
CANDIDATES = [
    os.path.abspath(os.path.join(HERE, "..")),        # works for Home.py
    os.path.abspath(os.path.join(HERE, "..", "..")),  # works for pages/*
]
for ROOT in CANDIDATES:
    if os.path.isdir(os.path.join(ROOT, "core")) and ROOT not in sys.path:
        sys.path.insert(0, ROOT)
        break

import streamlit as st

st.set_page_config(page_title="BioVis Studio", layout="wide")

st.title("🧬 BioVis Studio")
st.write(
    "Upload a dataset → pick a preprocessing recipe → run PCA/UMAP → make plots → export a reproducible notebook."
)

st.markdown("Use the left sidebar to navigate pages: Import → Preprocess → Dimensionality Reduction → Visualize → Export.")

st.markdown("---")
st.subheader("Tips")
st.markdown(
    "- Use tidy CSV/TSV (samples × features).\n"
    "- Include sample-level metadata columns (e.g., Group, Batch).\n"
    "- For RNA-seq counts, try the RNA-seq preset in Preprocess."
)
