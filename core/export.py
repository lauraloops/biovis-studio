import os
from datetime import datetime
from pathlib import Path
import nbformat as nbf

def export_notebook(state, out_dir: str | None = None) -> str:
    """
    Render a lightweight reproducibility notebook and write it to disk.
    Returns the full file path.
    """
    # Choose a safe, user-writable export directory
    if out_dir is None:
        out_dir = os.path.join(os.getcwd(), "exports")
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"biovis_run_{ts}.ipynb"
    out_path = os.path.join(out_dir, fname)

    # Minimal notebook with session state snapshot
    nb = nbf.v4.new_notebook()
    cells = []

    # Session summary cell
    keys = list(state.keys())
    cells.append(nbf.v4.new_markdown_cell(f"# BioVis Studio Run\n\nExported at {ts}\n\nSession keys: {', '.join(keys)}"))

    # Optional: include processed dataframe preview
    if "prep_meta" in state and hasattr(state["prep_meta"], "head"):
        df = state["prep_meta"]
        head_csv = df.head(20).to_csv(index=False)
        cells.append(nbf.v4.new_markdown_cell("## Data (head)"))
        cells.append(nbf.v4.new_code_cell(f"import pandas as pd\nfrom io import StringIO\ncsv = '''{head_csv}'''\ndf = pd.read_csv(StringIO(csv))\ndf"))

    nb["cells"] = cells

    # Write notebook
    nbf.write(nb, out_path)
    return out_path
