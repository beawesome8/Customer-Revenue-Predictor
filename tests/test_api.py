"""
API contract tests - run in CI on every PR, before the regression gate,
because a broken schema is a faster and cheaper failure to catch than a
full retrain. Fail fast on the cheap check first.
"""
import sys
sys.path.insert(0, ".")
from fastapi.testclient import TestClient
from src.serve.main import app

VALID_PAYLOAD = {
    "Administrative": 2, "Administrative_Duration": 45.0,
    "Informational": 0, "Informational_Duration": 0.0,
    "ProductRelated": 15, "ProductRelated_Duration": 600.0,
    "BounceRates": 0.02, "ExitRates": 0.05, "PageValues": 12.5,
    "SpecialDay": 0.0, "Month": "Nov", "OperatingSystems": 2,
    "Browser": 2, "Region": 1, "TrafficType": 2,
    "VisitorType": "Returning_Visitor", "Weekend": False,
}


def test_health_endpoint():
    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["model_loaded"] is True


def test_predict_valid_payload():
    with TestClient(app) as client:
        r = client.post("/predict", json=VALID_PAYLOAD)
        assert r.status_code == 200
        body = r.json()
        assert 0.0 <= body["purchase_probability"] <= 1.0
        assert isinstance(body["will_purchase"], bool)


def test_predict_rejects_invalid_month():
    with TestClient(app) as client:
        bad = dict(VALID_PAYLOAD, Month="Smarch")
        r = client.post("/predict", json=bad)
        assert r.status_code == 422


def test_predict_rejects_out_of_range_bounce_rate():
    with TestClient(app) as client:
        bad = dict(VALID_PAYLOAD, BounceRates=1.5)
        r = client.post("/predict", json=bad)
        assert r.status_code == 422


def test_predict_rejects_missing_field():
    with TestClient(app) as client:
        bad = dict(VALID_PAYLOAD)
        del bad["PageValues"]
        r = client.post("/predict", json=bad)
        assert r.status_code == 422
