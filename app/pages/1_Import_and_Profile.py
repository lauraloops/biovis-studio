import os, sys, tempfile
import streamlit as st
import pandas as pd
import numpy as np

HERE = os.path.dirname(__file__)
CANDIDATES = [
    os.path.abspath(os.path.join(HERE, "..")),
    os.path.abspath(os.path.join(HERE, "..", "..")),
]
for ROOT in CANDIDATES:
    if os.path.isdir(os.path.join(ROOT, "core")) and ROOT not in sys.path:
        sys.path.insert(0, ROOT)
        break

from core.io import read_any, infer_schema, detect_id_columns, detect_duplicate_ids, infer_orientation

st.set_page_config(page_title="Import & Profile", layout="wide")
st.header("1️⃣ Import & Profile Data")

uploaded = st.file_uploader(
    "📤 Upload CSV / TSV / Parquet",
    type=["csv", "tsv", "txt", "parquet"]
)

def _safe_to_temp(uploaded_file) -> str:
    # Persist the uploaded file to a temp path to allow efficient file IO
    suffix = os.path.splitext(uploaded_file.name)[1].lower()
    fd, tmp_path = tempfile.mkstemp(prefix="biovis_", suffix=suffix)
    with os.fdopen(fd, "wb") as out:
        out.write(uploaded_file.read())
    return tmp_path

def _read_large_csv(path: str, sep: str | None = None, max_rows: int | None = None) -> pd.DataFrame:
    # Detect delimiter if not provided
    if sep is None:
        if path.endswith(".tsv") or path.endswith(".txt"):
            sep = "\t"
        else:
            sep = ","
    # Estimate total rows for progress (optional quick pass)
    total_rows = None
    try:
        with open(path, "rb") as f:
            total_rows = sum(1 for _ in f) - 1  # exclude header
    except Exception:
        pass

    chunks = []
    chunksize = 250_000  # tune for memory; 250k rows per chunk
    rows_read = 0
    prog = st.progress(0, text="Reading CSV in chunks…")
    for i, chunk in enumerate(pd.read_csv(
        path,
        sep=sep,
        chunksize=chunksize,
        low_memory=True,
        engine="c",
    )):
        chunks.append(chunk)
        rows_read += len(chunk)
        if total_rows:
            prog.progress(min(1.0, rows_read / max(total_rows, 1.0)), text=f"Reading… {rows_read:,}/{total_rows:,} rows")
        else:
            prog.progress(0.0, text=f"Reading… {rows_read:,} rows")
        if max_rows and rows_read >= max_rows:
            break
    prog.empty()
    df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
    return df

def _read_parquet(path: str) -> pd.DataFrame:
    # PyArrow-backed reader is fast and memory efficient
    return pd.read_parquet(path)

if uploaded:
    st.info("Large file detected? We store to a temp file and parse in chunks for stability.")
    tmp_path = _safe_to_temp(uploaded)

    try:
        if uploaded.name.lower().endswith((".parquet", ".pq")):
            df = _read_parquet(tmp_path)
        else:
            df = _read_large_csv(tmp_path)

        st.session_state["raw_df"] = df
        st.session_state["working_df"] = df.copy()

        st.success(f"✅ Loaded {len(df):,} rows × {len(df.columns):,} columns from {uploaded.name}")
        st.caption(f"Temp path: {tmp_path}")

        tab_profile, tab_structure, tab_metadata = st.tabs(
            ["📊 Profile", "🔧 Structure & Orientation", "🏷️ Metadata Columns"]
        )
        with tab_profile:
            st.dataframe(df.head(100), use_container_width=True)
            st.write(df.describe(include="all").transpose())

        with tab_structure:
            st.write("Infer orientation and structure in Data Preparation.")

        with tab_metadata:
            st.write("Select metadata columns in Data Preparation.")

    except Exception as e:
        st.error(f"❌ Failed to read file: {e}")
        st.exception(e)
    finally:
        # Optional: keep temp for debugging; otherwise uncomment to remove
        # os.remove(tmp_path)
        pass
else:
    st.warning("Upload a CSV/TSV or Parquet file to begin.")
