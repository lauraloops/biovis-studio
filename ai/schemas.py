from typing import List, Dict, Any
from pydantic import BaseModel


class ClusterSummary(BaseModel):
    label: str
    size: int
    dominant_metadata: Dict[str, Any] | None = None


class AIInsightResult(BaseModel):
    summary: str
    clusters: List[ClusterSummary]
    outliers: List[int]
    hypotheses: List[str]
    next_steps: List[str]
    caveats: List[str]
