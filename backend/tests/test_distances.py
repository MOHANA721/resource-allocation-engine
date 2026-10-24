"""Distance sources: OSRM, detour, and straight line."""

import pytest
from fastapi.testclient import TestClient

from app import distances
from app.allocate import allocate
from app.distances import DETOUR_FACTOR, DistanceTable, haversine_km, point
from app.generator import generate, trap_scenario
from app.main import app
from conftest import FAKE_ROAD_FACTOR

client = TestClient(app)
ALGORITHMS = ("greedy", "hungarian", "regret")


def _pairs(result):
    return {row.request_id: row.technician_id for row in result.assignments}


def test_detour_scales_every_distance():
    scenario = trap_scenario()
    straight = allocate(scenario.technicians, scenario.requests, distance="straight")
    detour = allocate(scenario.technicians, scenario.requests, distance="detour")
    assert straight.distance_source == "straight"
    assert detour.distance_source == "detour"
    for name in ALGORITHMS:
        assert _pairs(getattr(detour.results, name)) == _pairs(getattr(straight.results, name))
        assert getattr(detour.results, name).metrics.total_km == pytest.approx(
            DETOUR_FACTOR * getattr(straight.results, name).metrics.total_km
        )


def test_osrm_distances_are_used_when_available(fake_osrm):
    scenario = trap_scenario()
    straight = allocate(scenario.technicians, scenario.requests, distance="straight")
    road = allocate(scenario.technicians, scenario.requests, distance="osrm")
    assert road.distance_source == "osrm"
    assert "Road distances from OSRM" in road.distance_note
    assert fake_osrm.calls == [4]
    assert road.results.hungarian.metrics.total_km == pytest.approx(
        FAKE_ROAD_FACTOR * straight.results.hungarian.metrics.total_km
    )


def test_unreachable_osrm_falls_back_to_detour(fake_osrm):
    fake_osrm.fail = "timed out"
    scenario = trap_scenario()
    outcome = allocate(scenario.technicians, scenario.requests, distance="osrm")
    assert outcome.distance_source == "detour"
    assert "unavailable (timed out)" in outcome.distance_note
    detour = allocate(scenario.technicians, scenario.requests, distance="detour")
    assert outcome.results.hungarian.metrics.total_km == pytest.approx(detour.results.hungarian.metrics.total_km)


def test_large_scenarios_are_fetched_in_blocks_and_cached(fake_osrm):
    scenario = generate(40, 80, seed=3, scarcity=0.4)
    allocate(scenario.technicians, scenario.requests, distance="osrm")
    assert len(fake_osrm.calls) == 3
    assert max(fake_osrm.calls) <= distances.OSRM_MAX_COORDINATES
    points = sorted({point(t.lat, t.lng) for t in scenario.technicians} | {point(r.lat, r.lng) for r in scenario.requests})
    assert all((a, b) in distances._cache for a in points for b in points if a != b)
    allocate(scenario.technicians, scenario.requests, distance="osrm")
    assert len(fake_osrm.calls) == 3


def test_unroutable_pair_falls_back_to_detour(monkeypatch):
    def fetch(points):
        return [[None if i == 0 and j == 1 else 2.0 for j in range(len(points))] for i in range(len(points))]

    monkeypatch.setattr(distances, "_fetch", fetch)
    a, b = point(12.97, 77.59), point(12.98, 77.60)
    table = distances.build([a, b], "osrm")
    assert table.km(b, a) == 2.0
    assert table.km(a, b) == pytest.approx(haversine_km(*a, *b) * DETOUR_FACTOR)
    assert isinstance(table, DistanceTable)


def test_api_defaults_to_osrm_and_validates_choices(fake_osrm):
    scenario = client.get("/api/scenario/trap").json()
    body = {"technicians": scenario["technicians"], "requests": scenario["requests"]}
    default = client.post("/api/allocate", json=body).json()
    assert default["distance_source"] == "osrm"
    assert fake_osrm.calls
    calls = len(fake_osrm.calls)
    straight = client.post("/api/allocate", json={**body, "distance": "straight"}).json()
    assert straight["distance_source"] == "straight"
    assert len(fake_osrm.calls) == calls
    assert client.post("/api/allocate", json={**body, "distance": "road"}).status_code == 422
