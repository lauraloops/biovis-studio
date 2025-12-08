import os
from datetime import datetime
import nbformat as nbf

def export_notebook(state: dict) -> str:
    nb = nbf.v4.new_notebook()
    cells = []
    cells.append(nbf.v4.new_markdown_cell("# BioVis Studio — Reproducible Run"))
    cells.append(nbf.v4.new_code_cell("import pandas as pd\nimport numpy as np\nfrom sklearn.decomposition import PCA"))
    # Minimal: you can serialize parameters from state
    cells.append(nbf.v4.new_markdown_cell("Parameters and steps would be injected here."))
    nb["cells"] = cells
    out = f"/mnt/data/biovis_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.ipynb"
    nbf.write(nb, out)
    return out
