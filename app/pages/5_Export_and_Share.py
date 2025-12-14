import os
import streamlit as st
from core.export import export_notebook

st.set_page_config(page_title="5) Export & Share", layout="wide")
st.header("5) Export & Share")

# Choose export directory (user-writable)
default_dir = os.path.join(os.getcwd(), "exports")
out_dir = st.text_input("Export directory", value=default_dir)

if st.button("Generate Reproducibility Notebook"):
    try:
        nb_path = export_notebook(st.session_state, out_dir=out_dir)
        st.success(f"Notebook created at: {nb_path}")
        with open(nb_path, "rb") as f:
            st.download_button(
                label="Download notebook (.ipynb)",
                data=f.read(),
                file_name=os.path.basename(nb_path),
                mime="application/x-ipynb+json"
            )
    except Exception as e:
        st.error(f"Failed to export notebook: {e}")
        st.exception(e)
