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
import pandas as pd
from core.viz import make_scatter

st.header("4) Visualize Studio")

if "prep_meta" not in st.session_state:
    st.warning("You need an embedding or PCA scores; run Dimensionality Reduction first.")
    st.stop()

meta = st.session_state["prep_meta"]
cols = list(meta.columns)

color_by = st.selectbox("Color by", [None] + cols, index=0)
shape_by = st.selectbox("Shape by", [None] + cols, index=0)
size_candidates = [c for c in cols if pd.api.types.is_numeric_dtype(meta[c])]
size_by = st.selectbox("Size by (numeric)", [None] + size_candidates, index=0)

fig = make_scatter(meta, color_by=color_by, shape_by=shape_by, size_by=size_by)
st.plotly_chart(fig, use_container_width=True)
