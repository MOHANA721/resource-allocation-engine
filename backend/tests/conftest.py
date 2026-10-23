"""Tests never call the real OSRM server. The fake returns straight-line km × 1.3."""

import pytest

from app import distances

FAKE_ROAD_FACTOR = 1.3


class FakeOsrm:
    def __init__(self):
        self.calls: list[int] = []
        self.fail: str | None = None

    def __call__(self, points):
        if self.fail:
            raise distances.OsrmError(self.fail)
        assert len(points) <= distances.OSRM_MAX_COORDINATES
        self.calls.append(len(points))
        return [[distances.haversine_km(*a, *b) * FAKE_ROAD_FACTOR for b in points] for a in points]


@pytest.fixture(autouse=True)
def fake_osrm(monkeypatch):
    fake = FakeOsrm()
    monkeypatch.setattr(distances, "_fetch", fake)
    distances._cache.clear()
    yield fake
    distances._cache.clear()
