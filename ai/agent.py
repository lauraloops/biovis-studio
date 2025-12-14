# core/ai/agent.py
import plotly.io as pio
from .llm_gemini import GeminiClient
from .context_builder import build_context
from .schemas import AIInsightResult
import json
import re


def fig_to_png_bytes(fig) -> bytes:
    return fig.to_image(format="png", scale=2)


SYSTEM_PROMPT = """
You are an expert in biological and biomedical visual analytics.

You receive:
- A scatter plot image from PCA / UMAP / t-SNE
- A table containing the exact plotted points (id, x, y, metadata)
- Context about the dataset and embedding method

Rules:
- Use the table as ground truth for counts and outliers
- Use the image to describe structure and patterns
- Be cautious: if evidence is insufficient, say so
- Avoid speculative biological claims

Output TWO parts:
1) A short readable explanation
2) A JSON object following this schema:

{
  "summary": string,
  "clusters": [{ "label": string, "size": number, "dominant_metadata": object | null }],
  "outliers": [number],
  "hypotheses": [string],
  "next_steps": [string],
  "caveats": [string]
}

Return valid JSON only in part (2).
""".strip()


def extract_json(text: str) -> dict | None:
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except Exception:
        return None


def analyze(state: dict, question: str, model: str = "gemini-2.5-flash"):
    if "viz_fig" not in state:
        raise ValueError("state must contain 'viz_fig'")

    client = GeminiClient(model=model)
    image_bytes = fig_to_png_bytes(state["viz_fig"])
    context = build_context(state)

    prompt = f"""
{SYSTEM_PROMPT}

User question:
{question}

Context:
{context}
""".strip()

    response_text = client.generate(prompt, image_bytes)

    parsed = extract_json(response_text)
    parsed_obj = None
    if parsed:
        try:
            parsed_obj = AIInsightResult(**parsed)
        except Exception:
            parsed_obj = None

    return {
        "text": response_text,
        "parsed": parsed_obj,
    }
