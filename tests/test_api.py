from fastapi.testclient import TestClient
import pytest
from unittest.mock import Mock, patch
from phenocam.data.api import app

@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture
def mock_vector_store():
    with patch('phenocam.data.api.vector_store') as mock:
        mock.similar.return_value = ["url1.jpg", "url2.jpg"]
        mock.labelled.return_value = ["url3.jpg", "url4.jpg"]
        yield mock

@pytest.fixture
def sample_embeddings():
    return {"embeddings": [0.1, 0.2, 0.3], "n_results": 2}

def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_query_similar(client, mock_vector_store, sample_embeddings):
    response = client.post("/query/similar", json=sample_embeddings)