"""Correctness and comparison tests for the multi-job field service engine."""

import pytest
from fastapi.testclient import TestClient

from app.allocate import UnofferedSkill, allocate
from app.constraints import check
from app.generator import generate, trap_scenario
from app.main import app
from app.models import Request, Technician, Weights

client = TestClient(app)


def _pairs(result):
    return {row.request_id: row.technician_id for row in result.assignments}


def _by_id(scenario):
    techs = {tech.id: tech for tech in scenario.technicians}
    requests = {request.id: request for request in scenario.requests}
    return techs, requests


def test_same_seed_is_identical():
    first = generate(10, 20, seed=1, scarcity=0.4)
    second = generate(10, 20, seed=1, scarcity=0.4)
    assert first.model_dump() == second.model_dump()


def test_empty_request_list():
    scenario = generate(5, 0, seed=2, scarcity=0.2)
    result = allocate(scenario.technicians, scenario.requests)
    assert result.results.greedy.assignments == []
    assert result.results.greedy.metrics.priority_coverage == 1.0


def test_hard_constraints_hold_across_seeds():
    for seed in range(25):
        scenario = generate(10, 18, seed=seed, scarcity=0.5)
        techs, requests = _by_id(scenario)
        outcome = allocate(scenario.technicians, scenario.requests)
        for name in ("greedy", "hungarian", "regret"):
            result = getattr(outcome.results, name)
            assert {row.request_id for row in result.assignments} == set(requests)
            load: dict[str, int] = {}
            for row in result.assignments:
                if row.technician_id is None:
                    assert row.explanation
                    continue
                tech = techs[row.technician_id]
                request = requests[row.request_id]
                assert request.required_skill in tech.skills
                assert check(tech, request, 0)[0]
                load[tech.id] = load.get(tech.id, 0) + 1
                assert load[tech.id] <= tech.max_jobs


def test_hungarian_is_never_worse_and_sometimes_better():
    for seed in range(60):
        scenario = generate(8, 16, seed=100 + seed, scarcity=0.7)
        outcome = allocate(scenario.technicians, scenario.requests)
        optimal = outcome.results.hungarian.metrics.total_cost
        assert optimal <= outcome.results.greedy.metrics.total_cost + 1e-6
        assert optimal <= outcome.results.regret.metrics.total_cost + 1e-6
    trap = allocate(trap_scenario().technicians, trap_scenario().requests)
    assert trap.results.hungarian.metrics.total_cost < trap.results.greedy.metrics.total_cost
    assert trap.results.hungarian.metrics.total_cost <= trap.results.regret.metrics.total_cost + 1e-6


def test_trap_greedy_strands_the_electrician():
    scenario = trap_scenario()
    outcome = allocate(scenario.technicians, scenario.requests)
    greedy = _pairs(outcome.results.greedy)
    hungarian = _pairs(outcome.results.hungarian)
    regret = _pairs(outcome.results.regret)
    assert greedy["leak"] == "vikram"
    assert greedy["panel"] is None
    assert hungarian["leak"] == "arjun"
    assert hungarian["panel"] == "vikram"
    assert regret["panel"] == "vikram"
    assert regret["leak"] == "arjun"
    assert outcome.results.greedy.metrics.high_priority_unassigned_count == 1
    assert outcome.results.hungarian.metrics.high_priority_unassigned_count == 0


def test_low_priority_is_dropped_when_capacity_is_short():
    tech = Technician(
        id="rahul",
        name="Rahul",
        lat=12.97,
        lng=77.59,
        area="Bengaluru",
        skills=["plumbing"],
        shift_start="08:00",
        shift_end="18:00",
        max_jobs=1,
    )
    urgent = Request(
        id="urgent",
        customer="Urgent",
        lat=12.971,
        lng=77.591,
        area="Bengaluru",
        required_skill="plumbing",
        window_start="09:00",
        window_end="12:00",
        duration=60,
        priority=5,
    )
    minor = Request(
        id="minor",
        customer="Minor",
        lat=12.972,
        lng=77.592,
        area="Bengaluru",
        required_skill="plumbing",
        window_start="09:00",
        window_end="15:00",
        duration=60,
        priority=1,
    )
    outcome = allocate([tech], [minor, urgent])
    for name in ("greedy", "hungarian", "regret"):
        pairs = _pairs(getattr(outcome.results, name))
        assert pairs["urgent"] == "rahul"
        assert pairs["minor"] is None


def test_a_request_for_an_unoffered_skill_is_rejected():
    tech = Technician(
        id="ravi",
        name="Ravi",
        lat=12.97,
        lng=77.59,
        area="Bengaluru",
        skills=["plumbing"],
        shift_start="08:00",
        shift_end="18:00",
        max_jobs=2,
    )
    gap = Request(
        id="ac",
        customer="AC household",
        lat=12.98,
        lng=77.60,
        area="Bengaluru",
        required_skill="hvac",
        window_start="09:00",
        window_end="12:00",
        duration=60,
        priority=4,
    )
    with pytest.raises(UnofferedSkill, match="hvac"):
        allocate([tech], [gap])
    response = client.post(
        "/api/allocate",
        json={"technicians": [tech.model_dump()], "requests": [gap.model_dump()], "distance": "straight"},
    )
    assert response.status_code == 422
    assert "hvac" in response.json()["detail"]


def test_abundant_resources_tie():
    techs = [
        Technician(
            id="rahul",
            name="Rahul",
            lat=12.9716,
            lng=77.5946,
            area="Bengaluru",
            skills=["plumbing"],
            shift_start="08:00",
            shift_end="18:00",
            max_jobs=4,
        ),
        Technician(
            id="vikram",
            name="Vikram",
            lat=12.9352,
            lng=77.6245,
            area="Koramangala",
            skills=["hvac"],
            shift_start="08:00",
            shift_end="18:00",
            max_jobs=4,
        ),
    ]
    requests = [
        Request(
            id="leak",
            customer="Leak",
            lat=12.9720,
            lng=77.5950,
            area="Bengaluru",
            required_skill="plumbing",
            window_start="09:00",
            window_end="12:00",
            duration=60,
            priority=3,
        ),
        Request(
            id="ac",
            customer="AC",
            lat=12.9360,
            lng=77.6250,
            area="Koramangala",
            required_skill="hvac",
            window_start="09:00",
            window_end="12:00",
            duration=60,
            priority=3,
        ),
    ]
    outcome = allocate(techs, requests)
    costs = [
        outcome.results.greedy.metrics.total_cost,
        outcome.results.hungarian.metrics.total_cost,
        outcome.results.regret.metrics.total_cost,
    ]
    assert max(costs) - min(costs) < 1e-6
    assert outcome.results.greedy.metrics.unassigned_count == 0


def test_travel_time_uses_forty_kmh():
    tech = Technician(
        id="suresh",
        name="Suresh",
        lat=12.97,
        lng=77.59,
        area="Bengaluru",
        skills=["plumbing"],
        shift_start="08:00",
        shift_end="17:00",
        max_jobs=1,
    )
    reachable = Request(
        id="job",
        customer="Household",
        lat=12.97 + 20 / 111,
        lng=77.59,
        area="Bengaluru",
        required_skill="plumbing",
        window_start="09:00",
        window_end="12:00",
        duration=120,
        priority=2,
    )
    too_early = reachable.model_copy(update={"id": "early", "window_end": "08:20"})
    assert check(tech, reachable, 0) == (True, "ok")
    assert check(tech, too_early, 0)[0] is False


def test_requests_only_ask_for_skills_the_workforce_has():
    for seed in range(100):
        scenario = generate(12, 30, seed=seed, scarcity=0.95)
        offered = {skill for tech in scenario.technicians for skill in tech.skills}
        assert offered
        assert {request.required_skill for request in scenario.requests} <= offered


def test_api_generate_trap_and_allocate():
    health = client.get("/api/health")
    assert health.status_code == 200
    created = client.post("/api/scenario", json={"n_techs": 8, "n_requests": 12, "seed": 3, "scarcity": 0.3})
    assert created.status_code == 200
    scenario = created.json()
    assert len(scenario["technicians"]) == 8
    trapped = client.get("/api/scenario/trap")
    assert trapped.status_code == 200
    allocated = client.post(
        "/api/allocate",
        json={"technicians": scenario["technicians"], "requests": scenario["requests"], "weights": Weights().model_dump()},
    )
    assert allocated.status_code == 200
    body = allocated.json()
    for name in ("greedy", "hungarian", "regret"):
        metrics = body["results"][name]["metrics"]
        assert "total_cost" in metrics
        assert "urgent_assigned_km" in metrics
        assert "workload_std" in metrics
