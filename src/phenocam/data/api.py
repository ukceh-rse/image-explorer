from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from phenocam.data.vectorstore import vector_store

app = FastAPI()


class SimilarityQuery(BaseModel):
    url: str
    n_results: int = 25


@app.get("/health")
async def health_check() -> dict:
    return {"status": "ok"}


@app.post("/query/similar")
async def query_similar(query: SimilarityQuery) -> dict:
    try:
        embeddings = vector_store.get(query.url)
        results = vector_store.similar(embeddings=embeddings, n_results=query.n_results)
        return {"urls": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/query/label/{label}")
async def query_label(label: str, n_results: int = 50) -> dict:
    try:
        results = vector_store.labelled(label=label, n_results=n_results)
        return {"urls": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
