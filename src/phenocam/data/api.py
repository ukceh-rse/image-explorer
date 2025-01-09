import logging
from collections.abc import AsyncGenerator

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from phenocam.data.vectorstore import SQLiteVecStore, vector_store

logging.basicConfig(level=logging.DEBUG)

app = FastAPI()

# CORS boilerplate
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve up images
app.mount("/images", StaticFiles(directory="./data/images"), name="images")


def url_prefix(url: str) -> str:
    """
    Kludge. The images should have persistent URLs. This function
    takes filenames and adds a URL prefix.
    We do this here because it's easier to remove than done in the JS
    """
    year = url.split("_")[1][:4]
    return f"/images/{year}/{url}"


async def get_db() -> AsyncGenerator[SQLiteVecStore]:
    store = vector_store("sqlite", "./data/vectors/test.db")
    try:
        yield store
    finally:
        store.db.close()


class SimilarityQuery(BaseModel):
    url: str
    n_results: int = 25


@app.get("/health")
async def health_check() -> dict:
    return {"status": "ok"}


@app.get("/random")
async def random_image(db: SQLiteVecStore = Depends(get_db)) -> dict:
    try:
        url = db.random()
        return {"url": url_prefix(url)}
    except Exception as err:
        logging.info(err)
        raise HTTPException(status_code=500, detail=str(err))


@app.post("/query/similar")
async def query_similar(query: SimilarityQuery, db: SQLiteVecStore = Depends(get_db)) -> dict:
    try:
        embeddings = db.get(query.url)
    # TODO catch connection-specific errors
    except Exception as err:
        logging.debug(err)
        raise HTTPException(status_code=500, detail=str(err))

    if not embeddings:
        raise HTTPException(status_code=404, detail="URL not found in database")

    try:
        results = db.closest(embeddings=embeddings, n_results=query.n_results)
        return {"urls": [url_prefix(r) for r in results]}
    except Exception as err:
        logging.debug(err)
        raise HTTPException(status_code=500, detail=str(err))
