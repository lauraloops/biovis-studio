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
from core.export import export_notebook

st.header("5) Export & Share")

if "prep_meta" not in st.session_state:
    st.warning("Nothing to export yet.")
    st.stop()

nb_path = export_notebook(st.session_state)
st.success("Notebook generated.")
st.code(nb_path)
