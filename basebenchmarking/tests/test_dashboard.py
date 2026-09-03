"""
Unit tests for Dashboard Flask Server and REST API.
"""

import pytest
from dashboard.app import create_app


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_dashboard_index_route(client):
    res = client.get("/")
    assert res.status_code == 200
    assert b"Image Registration Benchmarking Platform" in res.data


def test_api_overview_route(client):
    res = client.get("/api/overview")
    assert res.status_code == 200
    data = res.get_json()

    assert "has_data" in data
    assert "dataset_size" in data
    assert "total_methods" in data


def test_api_methods_route(client):
    res = client.get("/api/methods")
    assert res.status_code == 200
    data = res.get_json()

    assert isinstance(data, list)
    method_ids = [m["id"] for m in data]
    assert "sift" in method_ids
    assert "akaze" in method_ids


def test_api_leaderboard_route(client):
    res = client.get("/api/leaderboard")
    assert res.status_code == 200
    data = res.get_json()

    assert "has_data" in data
    assert "leaderboard" in data


def test_api_isro_comparison_route(client):
    res = client.get("/api/isro-comparison")
    assert res.status_code == 200
    data = res.get_json()

    assert "metadata" in data
    assert "table" in data
    assert len(data["table"]) > 0  # Paper Table 3 data pre-loaded
