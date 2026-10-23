"""Optimality, every hard rule, weights, metrics, explanations, validation, and scale."""

import itertools
import math
import time

import pytest
from fastapi.testclient import TestClient

from app.allocate import allocate, skill_gaps
from app.constraints import check, pair_distance
from app.generator import generate, trap_scenario
from app.main import app
from app.models import Request, Technician, Weights

client = TestClient(app)
ALGORITHMS = ("greedy", "hungarian", "regret")


def _tech(tech_id, lat, lng, skills=("plumbing",), shift=("08:00", "18:00"), max_jobs=2):
    return Technician(
        id=tech_id,
        name=tech_id.title(),
        lat=lat,
        lng=lng,
        area="Bengaluru",
        skills=list(skills),
        shift_start=shift[0],
        shift_end=shift[1],
        max_jobs=max_jobs,
    )


def _request(request_id, lat, lng, skill="plumbing", window=("09:00", "12:00"), duration=60, priority=3):
    return Request(
        id=request_id,
        customer=request_id.title(),
        lat=lat,
        lng=lng,
        area="Bengaluru",
        required_skill=skill,
        window_start=window[0],
        window_end=window[1],
        duration=duration,
        priority=priority,
    )


def _pairs(result):
    return {row.request_id: row.technician_id for row in result.assignments}


def _brute_force_cost(techs, requests, weights):
    """Cheapest total over every possible plan, including leaving any request open."""
    best = math.inf
    choices = [None, *techs]
    for plan in itertools.product(choices, repeat=len(requests)):
        load: dict[str, int] = {}
        total = 0.0
        valid = True
        for tech, request in zip(plan, requests):
            if tech is None:
                total += weights.w_unassigned * request.priority
                continue
            if not check(tech, request, 0)[0]:
                valid = False
                break
            load[tech.id] = load.get(tech.id, 0) + 1
            if load[tech.id] > tech.max_jobs:
                valid = False
                break
            total += weights.w_distance * pair_distance(tech, request)
        if not valid:
            continue
        total += sum(weights.w_balance * held * (held - 1) / 2 for held in load.values())
        best = min(best, total)
    return best


@pytest.mark.parametrize("weights", [Weights(), Weights(w_distance=5, w_balance=2, w_unassigned=5)])
def test_hungarian_matches_brute_force_on_small_scenarios(weights):
    for seed in range(30):
        scenario = generate(3, 5, seed=seed, scarcity=0.5)
        live, _gaps = skill_gaps(scenario.technicians, scenario.requests)
        outcome = allocate(scenario.technicians, scenario.requests, weights)
        expected = _brute_force_cost(scenario.technicians, live, weights)
        assert outcome.results.hungarian.metrics.total_cost == pytest.approx(expected, abs=1e-6)


def test_job_must_finish_before_shift_ends():
    tech = _tech("ravi", 12.97, 77.59, shift=("08:00", "10:00"))
    fits = _request("fits", 12.979, 77.59, duration=60)
    too_long = _request("long", 12.979, 77.59, duration=120)
    assert check(tech, fits, 0) == (True, "ok")
    assert check(tech, too_long, 0) == (False, "shift")
    outcome = allocate([tech], [too_long])
    for name in ALGORITHMS:
        row = getattr(outcome.results, name).assignments[0]
        assert row.technician_id is None
        assert "finish inside his shift" in row.explanation


def test_early_arrival_waits_for_the_window():
    tech = _tech("ravi", 12.97, 77.59, shift=("08:00", "11:00"))
    later = _request("later", 12.979, 77.59, window=("10:00", "12:00"), duration=60)
    too_late_to_finish = _request("late", 12.979, 77.59, window=("10:30", "12:00"), duration=60)
    assert check(tech, later, 0) == (True, "ok")
    assert check(tech, too_late_to_finish, 0) == (False, "shift")


def test_capacity_rule_is_reported():
    tech = _tech("ravi", 12.97, 77.59, max_jobs=1)
    request = _request("job", 12.971, 77.59)
    assert check(tech, request, 1) == (False, "full")
    assert check(_tech("ravi", 12.97, 77.59, skills=("hvac",)), request, 0) == (False, "skill")


def test_trap_weights_that_undervalue_drops_change_only_hungarian():
    scenario = trap_scenario()
    default = allocate(scenario.technicians, scenario.requests, Weights())
    cheap_drop = allocate(scenario.technicians, scenario.requests, Weights(w_distance=5, w_balance=2, w_unassigned=5))
    assert _pairs(cheap_drop.results.hungarian) == {"leak": "vikram", "panel": None}
    assert _pairs(cheap_drop.results.regret) == {"leak": "arjun", "panel": "vikram"}
    assert _pairs(cheap_drop.results.greedy) == _pairs(default.results.greedy)
    assert _pairs(cheap_drop.results.regret) == _pairs(default.results.regret)
    assert cheap_drop.results.hungarian.metrics.total_cost < cheap_drop.results.regret.metrics.total_cost


def test_unassigned_weight_never_changes_greedy_or_regret_choices():
    for seed in range(10):
        scenario = generate(8, 20, seed=seed, scarcity=0.6)
        runs = [
            allocate(scenario.technicians, scenario.requests, Weights(w_unassigned=value))
            for value in (0.5, 30, 80)
        ]
        for name in ("greedy", "regret"):
            first = _pairs(getattr(runs[0].results, name))
            for run in runs[1:]:
                assert _pairs(getattr(run.results, name)) == first


def test_scaling_every_weight_keeps_every_plan():
    scenario = generate(10, 18, seed=4, scarcity=0.4)
    base = allocate(scenario.technicians, scenario.requests, Weights(w_distance=1, w_balance=2, w_unassigned=30))
    scaled = allocate(scenario.technicians, scenario.requests, Weights(w_distance=3, w_balance=6, w_unassigned=90))
    for name in ALGORITHMS:
        assert _pairs(getattr(scaled.results, name)) == _pairs(getattr(base.results, name))
        assert getattr(scaled.results, name).metrics.total_cost == pytest.approx(
            3 * getattr(base.results, name).metrics.total_cost
        )


def test_balance_weight_spreads_jobs_in_every_algorithm():
    techs = [_tech("a", 12.97, 77.59), _tech("b", 12.97, 77.60)]
    requests = [_request("r1", 12.97, 77.591), _request("r2", 12.971, 77.59)]
    packed = allocate(techs, requests, Weights(w_balance=0))
    spread = allocate(techs, requests, Weights(w_balance=10))
    for name in ALGORITHMS:
        assert set(_pairs(getattr(packed.results, name)).values()) == {"a"}
        assert set(_pairs(getattr(spread.results, name)).values()) == {"a", "b"}


def test_hungarian_fills_a_technicians_cheaper_slot_first():
    techs = [_tech("a", 12.97, 77.59), _tech("b", 12.97, 77.65)]
    requests = [_request("r1", 12.97, 77.591), _request("r2", 12.971, 77.59)]
    outcome = allocate(techs, requests, Weights(w_balance=2))
    rows = outcome.results.hungarian.assignments
    assert {row.technician_id for row in rows} == {"a"}
    balance_paid = sorted(round(row.cost - row.distance_km, 6) for row in rows)
    assert balance_paid == [0.0, 2.0]


def test_trap_metrics_match_hand_calculation():
    scenario = trap_scenario()
    techs = {tech.id: tech for tech in scenario.technicians}
    requests = {request.id: request for request in scenario.requests}
    vikram_leak = pair_distance(techs["vikram"], requests["leak"])
    arjun_leak = pair_distance(techs["arjun"], requests["leak"])
    vikram_panel = pair_distance(techs["vikram"], requests["panel"])
    outcome = allocate(scenario.technicians, scenario.requests)

    hungarian = outcome.results.hungarian.metrics
    assert hungarian.total_km == pytest.approx(arjun_leak + vikram_panel)
    assert hungarian.avg_km == pytest.approx((arjun_leak + vikram_panel) / 2)
    assert hungarian.total_cost == pytest.approx(arjun_leak + vikram_panel)
    assert hungarian.priority_coverage == 1.0
    assert hungarian.workload_std == 0.0
    assert hungarian.urgent_assigned_km == pytest.approx(hungarian.total_km)
    assert hungarian.optimality_gap == 0.0

    greedy = outcome.results.greedy.metrics
    assert greedy.total_cost == pytest.approx(vikram_leak + 150)
    assert greedy.priority_coverage == 0.5
    assert greedy.workload_std == pytest.approx(0.5)
    assert greedy.urgent_avg_km == pytest.approx(vikram_leak)
    assert greedy.optimality_gap == pytest.approx(greedy.total_cost - hungarian.total_cost)


def test_explanations_name_the_real_reason():
    scenario = trap_scenario()
    default = allocate(scenario.technicians, scenario.requests)
    greedy = {row.request_id: row.explanation for row in default.results.greedy.assignments}
    hungarian = {row.request_id: row.explanation for row in default.results.hungarian.assignments}
    regret = {row.request_id: row.explanation for row in default.results.regret.assignments}
    assert "Runner-up Arjun" in greedy["leak"]
    assert "every technician with electrical is already full" in greedy["panel"]
    assert "Vikram is closer" in hungarian["leak"]
    assert "kept for another request" in hungarian["leak"]
    assert "largest regret" in regret["panel"]

    cheap_drop = allocate(scenario.technicians, scenario.requests, Weights(w_distance=5, w_balance=2, w_unassigned=5))
    dropped = next(row for row in cheap_drop.results.hungarian.assignments if row.request_id == "panel")
    assert "left open in the minimum-cost plan" in dropped.explanation
    assert "25.0" in dropped.explanation
    assert "already full" not in dropped.explanation


def test_skill_gap_inside_a_normal_scenario():
    scenario = trap_scenario()
    gap = _request("ac", 12.94, 77.62, skill="hvac", priority=5)
    plain = allocate(scenario.technicians, scenario.requests)
    mixed = allocate(scenario.technicians, [*scenario.requests, gap])
    assert [item.request_id for item in mixed.skill_gaps] == ["ac"]
    assert "nobody on this workforce has hvac" in mixed.skill_gaps[0].reason
    for name in ALGORITHMS:
        result = getattr(mixed.results, name)
        assert len(result.assignments) == 2
        assert _pairs(result) == _pairs(getattr(plain.results, name))
        assert result.metrics.total_cost == pytest.approx(getattr(plain.results, name).metrics.total_cost)


def test_algorithms_are_repeatable():
    scenario = generate(12, 20, seed=7, scarcity=0.4)
    first = allocate(scenario.technicians, scenario.requests)
    second = allocate(scenario.technicians, scenario.requests)
    for name in ALGORITHMS:
        a = getattr(first.results, name)
        b = getattr(second.results, name)
        assert [row.model_dump() for row in a.assignments] == [row.model_dump() for row in b.assignments]
        assert a.metrics.total_cost == b.metrics.total_cost


@pytest.mark.parametrize(
    "body",
    [
        {"n_techs": 0, "n_requests": 5, "seed": 1, "scarcity": 0.4},
        {"n_techs": 41, "n_requests": 5, "seed": 1, "scarcity": 0.4},
        {"n_techs": 5, "n_requests": 81, "seed": 1, "scarcity": 0.4},
        {"n_techs": 5, "n_requests": 5, "seed": 1, "scarcity": 1.5},
    ],
)
def test_api_rejects_bad_scenario_settings(body):
    assert client.post("/api/scenario", json=body).status_code == 422


def test_api_rejects_bad_allocate_input():
    scenario = client.get("/api/scenario/trap").json()
    bad_priority = [dict(scenario["requests"][0], priority=6)]
    response = client.post("/api/allocate", json={"technicians": scenario["technicians"], "requests": bad_priority})
    assert response.status_code == 422
    no_skills = [dict(scenario["technicians"][0], skills=[])]
    response = client.post("/api/allocate", json={"technicians": no_skills, "requests": scenario["requests"]})
    assert response.status_code == 422
    negative = {"w_distance": -1, "w_balance": 2, "w_unassigned": 30}
    response = client.post(
        "/api/allocate",
        json={"technicians": scenario["technicians"], "requests": scenario["requests"], "weights": negative},
    )
    assert response.status_code == 422


def test_largest_slider_setting_stays_fast_and_valid():
    scenario = generate(40, 80, seed=3, scarcity=0.4)
    techs = {tech.id: tech for tech in scenario.technicians}
    started = time.perf_counter()
    outcome = allocate(scenario.technicians, scenario.requests)
    assert time.perf_counter() - started < 10
    for name in ALGORITHMS:
        load: dict[str, int] = {}
        for row in getattr(outcome.results, name).assignments:
            if row.technician_id is None:
                continue
            load[row.technician_id] = load.get(row.technician_id, 0) + 1
            assert load[row.technician_id] <= techs[row.technician_id].max_jobs
    assert outcome.results.hungarian.metrics.total_cost <= outcome.results.greedy.metrics.total_cost + 1e-6
    assert outcome.results.hungarian.metrics.total_cost <= outcome.results.regret.metrics.total_cost + 1e-6
