from fastapi import FastAPI
from api.routers import datasets, runs

app = FastAPI(title="BioVis API")

@app.get("/health")
def health():
    return {"ok": True}

app.include_router(datasets.router, prefix="/datasets", tags=["datasets"])
app.include_router(runs.router, prefix="/runs", tags=["runs"])
