# core/ai/context_builder.py
import pandas as pd


def summarize_points(points_df: pd.DataFrame, max_rows: int = 2000) -> str:
    df = points_df.copy()
    if len(df) > max_rows:
        df = df.sample(max_rows, random_state=42)
        header = f"(Sampled {max_rows} of {len(points_df)} points)\n"
    else:
        header = f"(Full set: {len(df)} points)\n"

    return header + df.to_csv(index=False)


def build_context(state: dict) -> str:
    parts = []

    if "raw_df" in state and state["raw_df"] is not None:
        df = state["raw_df"]
        parts.append(f"Dataset shape: {df.shape}")
        parts.append("Column dtypes: " + str(df.dtypes.astype(str).value_counts().to_dict()))
        parts.append("First columns: " + ", ".join(df.columns[:10]))

    if state.get("dr_method"):
        parts.append(f"Dimensionality reduction: {state['dr_method']}")
    if state.get("dr_params"):
        parts.append(f"DR parameters: {state['dr_params']}")
    if state.get("dr_metrics"):
        parts.append(f"Embedding quality metrics: {state['dr_metrics']}")

    if state.get("viz_config"):
        parts.append(f"Plot encodings: {state['viz_config']}")

    if state.get("viz_points_df") is not None:
        parts.append("Plotted points table:")
        parts.append(summarize_points(state["viz_points_df"]))

    return "\n\n".join(parts)
