# BioVis Studio (MVP)

A minimal visual analytics workbench built with **Streamlit + Python**.

## Features
- Upload CSV/TSV/Parquet
- Schema profiling
- Preprocessing recipes (log1p, scaling, imputation, one-hot, RNA-seq preset)
- PCA (scores, scree, loadings)
- UMAP / t-SNE (basic params)
- Plot studio (color/shape/size)
- Export notebook (nbformat)

## Quickstart
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app/Home.py
# (optional API) uvicorn api.main:app --reload
```
