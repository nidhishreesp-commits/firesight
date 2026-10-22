import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    from firesight.api.main import app
    return TestClient(app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_testsets(client):
    r = client.get("/api/testsets")
    assert r.status_code == 200
    assert "test_bench" in r.json()


def test_testset_images(client):
    r = client.get("/api/testsets/test_mini/images?limit=2&threshold=0.5")
    assert r.status_code == 200
    body = r.json()
    assert body["total"] > 0
    assert len(body["items"]) == 2


def test_feedback_roundtrip(client):
    r = client.post("/api/feedback", json={"record_id": "unit-test", "verdict": "false_positive"})
    assert r.status_code == 200
    queue = client.get("/api/review/queue").json()
    assert any(row["record_id"] == "unit-test" for row in queue["items"])


def test_detect_image(client):
    import cv2
    import numpy as np
    img = (np.random.default_rng(1).random((120, 120, 3)) * 255).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    r = client.post("/api/detect/image", files={"file": ("x.jpg", buf.tobytes(), "image/jpeg")}, data={"threshold": "0.5"})
    assert r.status_code == 200
    assert "detections" in r.json()


def test_metrics(client):
    assert client.get("/metrics").status_code == 200


def test_fire_type_policy_roundtrip(client):
    r = client.get("/api/policy/fire-types")
    assert r.status_code == 200
    old = r.json()
    new = {"include": ["uncontrolled", "controlled", "unknown"], "exclude": ["spark", "screen"]}
    assert client.post("/api/policy/fire-types", json=new).status_code == 200
    current = client.get("/api/policy/fire-types").json()
    assert set(current["include"]) == set(new["include"])
    assert set(current["exclude"]) == set(new["exclude"])
    client.post("/api/policy/fire-types", json=old)


def test_bench100_has_100_images_and_lock_is_guarded(client):
    r = client.get("/api/bench100")
    assert r.status_code == 200
    body = r.json()
    assert len(body["images"]) == 100
    if body["locked"]:
        assert body["lock"]["manifest_sha256"]
    else:
        lock = client.post("/api/bench100/lock?reviewer=pytest")
        assert lock.status_code in (409, 422)


def test_benchR_candidates_exist_and_are_not_locked(client):
    r = client.get("/api/benchR")
    assert r.status_code == 200
    body = r.json()
    assert len(body["images"]) == 1000
    assert body["composition"]["fire_only"] == 320
    assert body["composition"]["smoke_only"] == 100
    assert body["composition"]["both"] == 80
    assert body["composition"]["negative"] == 500
    assert body["locked"] is False
