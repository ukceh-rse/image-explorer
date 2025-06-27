from imagesearch.data.vectorstore import (
    vector_store,
    SQLiteVecStore,
    serialize_f32,
    deserialize,
)

import numpy as np
import pytest
import math
from datetime import datetime


@pytest.fixture
def temp_dir(tmp_path):
    """Creates a temporary directory using pytest's tmp_path fixture."""
    return tmp_path


def test_store(temp_dir):
    store = vector_store("sqlite", temp_dir / "test.db")  # default 'test_collection'
    filename = "https://example.com/filename.tif"
    store.add(
        url=filename,  # we use image location in s3 rather than text content
        embeddings=list(np.random.rand(2048)),
        date=datetime.now(),  # wants a list of lists
    )  # wants a list of ids

    record = store.get(filename)
    assert len(deserialize(record)) == 2048


def test_embeddings(temp_dir):
    STORE = temp_dir
    store = vector_store("sqlite", temp_dir / "test2.db")
    filename = "https://example.com/filename.tif"
    store.add(
        url=filename,
        embeddings=list(np.random.rand(2048)),
        date=datetime.now(),
        site="TEST",
    )
    total = store.embeddings()
    assert len(total)


@pytest.mark.parametrize("store_type", ["sqlite"])
def test_queries(store_type, temp_dir):
    store = vector_store(store_type, temp_dir / "test3.db")
    for i in range(0, 5):
        filename = f"https://example.com/filename{i}.tif"
        store.add(
            url=filename,
            embeddings=list(np.random.rand(2048)),
            date=datetime.now(),
            site="TEST",
        )

    sample = store.get("https://example.com/filename0.tif")
    close = store.closest(sample)
    #assert len(close)

    # Test more queries here as we've got the db set up
    ids = store.ids()
    assert len(ids)

    embeddings = store.embeddings()
    assert len(embeddings) == len(ids)


def test_sqlite_store(temp_dir):
    store = vector_store("sqlite", temp_dir / "test4.db")
    assert isinstance(store, SQLiteVecStore)
    filename = "https://example.com/filename.tif"
    store.add(
        url=filename,  # we use image location in s3 rather than text content
        embeddings=list(np.random.rand(2048)),
        date=datetime.now(),
        site="TEST",
    )
    embed = store.get(filename)
    assert embed


def test_closest_sqlite(temp_dir):
    store = vector_store("sqlite", temp_dir / "test5.db")
    for i in range(0, 5):
        filename = f"https://example.com/filename{i}.tif"
        store.add(
            url=filename,  # we use http/s image location rather than text content
            embeddings=list(np.random.rand(2048)),
            date=datetime.now(),
            site="TEST",
        )

    sample = store.get("https://example.com/filename0.tif")
    close = store.closest(sample)
    #assert len(close)


def test_serialize_deserialize():
    """Round trip into compact format for sqlite-vec, back for working with floats"""

    for i in [2048, 512]:
        vec = tuple(np.random.rand(i))
        packed = serialize_f32(vec)
        vec2 = deserialize(packed)
        for j in range(0, i):
            assert math.isclose(vec[j], vec2[j], rel_tol=1e-07)
