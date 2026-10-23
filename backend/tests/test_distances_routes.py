"""Distance sources (OSRM, detour, straight line) and routed capacity."""

import pytest
from fastapi.testclient import TestClient

from app import distances
from app.allocate import allocate
from app.distances import DETOUR_FACTOR, DistanceTable, haversine_km, point
from app.generator import generate, trap_scenario
from app.main import app
from app.models import Request, Technician
from app.routes import route_km
from conftest import FAKE_ROAD_FACTOR

client = TestClient(app)
ALGORITHMS = ("greedy", "hungarian", "regret")
KM_EAST = 1 / (111.32 * 0.97449)


def _tech(tech_id="ravi", lat=12.97, lng=77.59, max_jobs=2, shift=("08:00", "18:00")):
    return Technician(
        id=tech_id, name=tech_id.title(), lat=lat, lng=lng, area="Bengaluru", skills=["plumbing"],
        shift_start=shift[0], shift_end=shift[1], max_jobs=max_jobs,
    )


def _request(request_id, lat, lng, window=("09:00", "17:00"), duration=60, priority=3):
    return Request(
        id=request_id, customer=request_id.title(), lat=lat, lng=lng, area="Bengaluru",
        required_skill="plumbing", window_start=window[0], window_end=window[1],
        duration=duration, priority=priority,
    )


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


def test_routed_mode_stops_overlapping_jobs():
    tech = _tech()
    first = _request("first", 12.971, 77.59, window=("09:00", "10:00"), duration=90)
    second = _request("second", 12.9712, 77.59, window=("09:00", "10:00"), duration=90)
    by_jobs = allocate([tech], [first, second])
    routed = allocate([tech], [first, second], capacity="routed")
    for name in ALGORITHMS:
        assert getattr(by_jobs.results, name).metrics.assigned_count == 2
        assert getattr(routed.results, name).metrics.assigned_count == 1
    for name in ("greedy", "regret"):
        missed = next(row for row in getattr(routed.results, name).assignments if row.technician_id is None)
        assert "fit it into his route" in missed.explanation
    missed = next(row for row in routed.results.hungarian.assignments if row.technician_id is None)
    assert "fully booked by job count or route timing" in missed.explanation


def test_routed_mode_drives_job_to_job():
    tech = _tech()
    near = _request("near", 12.97, 77.59 + KM_EAST)
    far = _request("far", 12.97, 77.59 + 2 * KM_EAST)
    by_jobs = allocate([tech], [near, far])
    routed = allocate([tech], [near, far], capacity="routed")
    for name in ALGORITHMS:
        assert getattr(by_jobs.results, name).metrics.total_km == pytest.approx(3.0, abs=0.01)
        result = getattr(routed.results, name)
        assert result.metrics.total_km == pytest.approx(2.0, abs=0.01)
        assert {row.request_id: row.stop for row in result.assignments} == {"near": 1, "far": 2}
        assert "to his route" in result.assignments[0].explanation


def test_routed_plans_are_drivable_across_seeds():
    for seed in range(15):
        scenario = generate(10, 30, seed=seed, scarcity=0.4)
        techs = {tech.id: tech for tech in scenario.technicians}
        requests = {request.id: request for request in scenario.requests}
        outcome = allocate(scenario.technicians, scenario.requests, capacity="routed")
        for name in ALGORITHMS:
            result = getattr(outcome.results, name)
            assert {row.request_id for row in result.assignments} == set(requests)
            routes: dict[str, list] = {}
            for row in result.assignments:
                if row.technician_id is not None:
                    routes.setdefault(row.technician_id, []).append(row)
            total = 0.0
            for tech_id, rows in routes.items():
                rows.sort(key=lambda row: row.stop)
                assert [row.stop for row in rows] == list(range(1, len(rows) + 1))
                assert len(rows) <= techs[tech_id].max_jobs
                length = route_km(techs[tech_id], [requests[row.request_id] for row in rows])
                assert length is not None
                assert length == pytest.approx(sum(row.distance_km for row in rows), abs=1e-6)
                total += length
            assert result.metrics.total_km == pytest.approx(total, abs=1e-6)


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
    routed = client.post("/api/allocate", json={**body, "capacity": "routed"})
    assert routed.status_code == 200
    assert {row["stop"] for row in routed.json()["results"]["hungarian"]["assignments"]} == {1}
