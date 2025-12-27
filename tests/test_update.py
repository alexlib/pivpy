import json

import pivpy


class _FakeResponse:
    def __init__(self, payload: bytes):
        self._payload = payload

    def read(self) -> bytes:
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_check_update_update_available(monkeypatch):
    monkeypatch.setattr(pivpy.update, "_get_installed_version_str", lambda *_a, **_k: "0.1.0")

    payload = json.dumps({"info": {"version": "0.2.0"}}).encode("utf-8")

    def fake_urlopen(_req, timeout=3.0):
        assert timeout == 3.0
        return _FakeResponse(payload)

    monkeypatch.setattr(pivpy.update.urllib.request, "urlopen", fake_urlopen)

    res = pivpy.check_update(timeout=3.0)
    assert res.status == 2
    assert res.installed == "0.1.0"
    assert res.latest == "0.2.0"


def test_check_update_up_to_date(monkeypatch):
    monkeypatch.setattr(pivpy.update, "_get_installed_version_str", lambda *_a, **_k: "1.2.3")

    payload = json.dumps({"info": {"version": "1.2.3"}}).encode("utf-8")

    monkeypatch.setattr(
        pivpy.update.urllib.request, "urlopen", lambda *_a, **_k: _FakeResponse(payload)
    )

    res = pivpy.check_update()
    assert res.status == 1
    assert res.latest == "1.2.3"


def test_check_update_server_unavailable(monkeypatch):
    monkeypatch.setattr(pivpy.update, "_get_installed_version_str", lambda *_a, **_k: "1.2.3")

    def boom(*_a, **_k):
        raise OSError("network down")

    monkeypatch.setattr(pivpy.update.urllib.request, "urlopen", boom)

    res = pivpy.check_update()
    assert res.status == 0
    assert res.installed == "1.2.3"
    assert res.latest == ""
