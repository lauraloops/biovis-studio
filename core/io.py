import pandas as pd

def read_any(file):
    name = getattr(file, "name", "uploaded")
    if name.endswith(".parquet"):
        return pd.read_parquet(file)
    if name.endswith(".tsv") or name.endswith(".txt"):
        return pd.read_csv(file, sep="\t")
    return pd.read_csv(file)

def infer_schema(df: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for c in df.columns:
        s = df[c]
        dtype = str(s.dtype)
        na_pct = float(s.isna().mean())
        uniq = int(s.nunique(dropna=True))
        parts.append({"column": c, "dtype": dtype, "na_pct": na_pct, "unique": uniq})
    return pd.DataFrame(parts)
