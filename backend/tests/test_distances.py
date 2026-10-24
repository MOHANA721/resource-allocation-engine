"""Distance sources: OSRM and straight line × 1.35."""

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


def _haversine_km(tech, request) -> float:
    return haversine_km(tech.lat, tech.lng, request.lat, request.lng)


def test_detour_scales_every_served_trip():
    scenario = trap_scenario()
    techs = {tech.id: tech for tech in scenario.technicians}
    requests = {request.id: request for request in scenario.requests}
    detour = allocate(scenario.technicians, scenario.requests, distance="detour")
    assert detour.distance_source == "detour"
    assert "× 1.35" in detour.distance_note
    for name in ALGORITHMS:
        for row in getattr(detour.results, name).assignments:
            if row.technician_id is None:
                continue
            assert row.distance_km == pytest.approx(
                DETOUR_FACTOR * _haversine_km(techs[row.technician_id], requests[row.request_id])
            )


def test_osrm_distances_are_used_when_available(fake_osrm):
    scenario = trap_scenario()
    techs = {tech.id: tech for tech in scenario.technicians}
    requests = {request.id: request for request in scenario.requests}
    road = allocate(scenario.technicians, scenario.requests, distance="osrm")
    assert road.distance_source == "osrm"
    assert "Road distances from OSRM" in road.distance_note
    assert fake_osrm.calls == [4]
    served = [row for row in road.results.hungarian.assignments if row.technician_id]
    expected = sum(
        FAKE_ROAD_FACTOR * _haversine_km(techs[row.technician_id], requests[row.request_id]) for row in served
    )
    assert road.results.hungarian.metrics.total_km == pytest.approx(expected)


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
    detour = client.post("/api/allocate", json={**body, "distance": "detour"}).json()
    assert detour["distance_source"] == "detour"
    assert len(fake_osrm.calls) == calls
    assert client.post("/api/allocate", json={**body, "distance": "straight"}).status_code == 422
    assert client.post("/api/allocate", json={**body, "distance": "road"}).status_code == 422
