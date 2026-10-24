"""Optimality, every hard rule, weights, metrics, explanations, validation, and scale."""

import itertools
import math
import time

import pytest
from fastapi.testclient import TestClient

from app.allocate import (
    _assignment_from_option,
    _finish,
    _Load,
    _open_or_decline,
    _options,
    _regret_key,
    allocate,
    run_regret,
)
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
    """Best (priority-5 misses, total cost) over every plan, including leaving any request open.

    Misses only count when priority 5 is protected.
    """
    best = (math.inf, math.inf)
    choices = [None, *techs]
    for plan in itertools.product(choices, repeat=len(requests)):
        load: dict[str, int] = {}
        total = 0.0
        missed_p5 = 0
        valid = True
        for tech, request in zip(plan, requests):
            if tech is None:
                total += weights.w_unassigned * request.priority
                missed_p5 += int(weights.must_serve_p5 and request.priority == 5)
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
        best = min(best, (missed_p5, total))
    return best


@pytest.mark.parametrize(
    "weights",
    [
        Weights(),
        Weights(w_distance=5, w_balance=2, w_unassigned=5),
        Weights(w_distance=5, w_balance=2, w_unassigned=5, must_serve_p5=False),
    ],
)
def test_hungarian_matches_brute_force_on_small_scenarios(weights):
    for seed in range(30):
        scenario = generate(3, 5, seed=seed, scarcity=0.5)
        outcome = allocate(scenario.technicians, scenario.requests, weights)
        expected_missed, expected_cost = _brute_force_cost(scenario.technicians, scenario.requests, weights)
        hungarian = outcome.results.hungarian
        missed = sum(1 for row in hungarian.assignments if row.technician_id is None and row.priority == 5)
        if weights.must_serve_p5:
            assert missed == expected_missed
        assert hungarian.metrics.total_cost == pytest.approx(expected_cost, abs=1e-6)


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


CHEAP_DROP = Weights(w_distance=5, w_balance=2, w_unassigned=5, must_serve_p5=False)


def test_trap_cheap_drops_change_only_hungarian_without_protection():
    scenario = trap_scenario()
    default = allocate(scenario.technicians, scenario.requests, Weights())
    cheap_drop = allocate(scenario.technicians, scenario.requests, CHEAP_DROP)
    assert _pairs(cheap_drop.results.hungarian) == {"leak": "vikram", "panel": None}
    assert _pairs(cheap_drop.results.regret) == {"leak": "arjun", "panel": "vikram"}
    assert _pairs(cheap_drop.results.greedy) == _pairs(default.results.greedy)
    assert _pairs(cheap_drop.results.regret) == _pairs(default.results.regret)
    assert cheap_drop.results.hungarian.metrics.total_cost < cheap_drop.results.regret.metrics.total_cost


def test_priority_five_protection_fixes_cheap_drop_weights():
    scenario = trap_scenario()
    protected = allocate(scenario.technicians, scenario.requests, CHEAP_DROP.model_copy(update={"must_serve_p5": True}))
    assert _pairs(protected.results.hungarian) == {"leak": "arjun", "panel": "vikram"}
    assert _pairs(protected.results.regret) == {"leak": "arjun", "panel": "vikram"}
    assert protected.results.hungarian.metrics.high_priority_unassigned_count == 0


def test_greedy_and_regret_leave_a_job_open_when_serving_costs_more():
    tech = _tech("ravi", 12.97, 77.59)
    far = _request("far", 12.97 + 25 / 111, 77.59, window=("09:00", "16:00"), priority=1)
    dropped = allocate([tech], [far], Weights(w_unassigned=10))
    for name in ALGORITHMS:
        assert _pairs(getattr(dropped.results, name)) == {"far": None}
    for name in ("greedy", "regret"):
        row = getattr(dropped.results, name).assignments[0]
        assert "left open on purpose" in row.explanation
        assert "penalty of 10.0" in row.explanation
    served = allocate([tech], [far], Weights(w_unassigned=30))
    for name in ALGORITHMS:
        assert _pairs(getattr(served.results, name)) == {"far": "ravi"}


def test_priority_five_is_served_even_when_the_penalty_is_small():
    tech = _tech("ravi", 12.97, 77.59)
    far = _request("far", 12.97 + 25 / 111, 77.59, window=("09:00", "16:00"), priority=5)
    protected = allocate([tech], [far], Weights(w_unassigned=1))
    unprotected = allocate([tech], [far], Weights(w_unassigned=1, must_serve_p5=False))
    for name in ALGORITHMS:
        assert _pairs(getattr(protected.results, name)) == {"far": "ravi"}
        assert _pairs(getattr(unprotected.results, name)) == {"far": None}


def test_unassigned_weight_now_changes_greedy_and_regret():
    for name in ("greedy", "regret"):
        changed = False
        for seed in range(10):
            scenario = generate(8, 20, seed=seed, scarcity=0.6)
            low, high = (
                allocate(scenario.technicians, scenario.requests, Weights(w_unassigned=value, must_serve_p5=False))
                for value in (0.5, 80)
            )
            low_open = getattr(low.results, name).metrics.unassigned_count
            high_open = getattr(high.results, name).metrics.unassigned_count
            assert low_open >= high_open
            changed = changed or low_open > high_open
        assert changed


def test_regret_ranks_priority_before_regret():
    techs = [_tech("a", 12.97, 77.59, max_jobs=1), _tech("b", 12.97, 77.70, max_jobs=1)]
    near_a = _request("x", 12.971, 77.59, window=("08:00", "08:10"), priority=1)
    near_b = _request("y", 12.971, 77.70, window=("08:00", "08:10"), priority=1)
    urgent = _request("z", 12.97, 77.645, window=("09:00", "12:00"), priority=5)
    outcome = allocate(techs, [near_a, near_b, urgent])
    regret = _pairs(outcome.results.regret)
    assert regret["z"] is not None
    assert outcome.results.regret.metrics.high_priority_unassigned_count == 0
    assert outcome.results.regret.metrics.unassigned_count == 1


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

    cheap_drop = allocate(scenario.technicians, scenario.requests, CHEAP_DROP)
    dropped = next(row for row in cheap_drop.results.hungarian.assignments if row.request_id == "panel")
    assert "left open in the minimum-cost plan" in dropped.explanation
    assert "25.0" in dropped.explanation
    assert "already full" not in dropped.explanation


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


def _regret_recomputed(techs, requests, weights):
    """Regret without the incremental update: every option is rebuilt every round."""
    load = _Load(techs)
    remaining = list(requests)
    decisions = {}
    while remaining:
        options = {request.id: _options(request, techs, load, weights) for request in remaining}
        settled = []
        for request in remaining:
            missed = _open_or_decline(request, options[request.id], techs, load, weights)
            if missed is not None:
                decisions[request.id] = missed
                settled.append(request.id)
        if settled:
            remaining = [request for request in remaining if request.id not in settled]
            continue
        request = min(remaining, key=lambda item: _regret_key(item, options[item.id]))
        _cost, _tech_id, tech, distance, jobs_held = options[request.id][0]
        decisions[request.id] = _assignment_from_option(
            request, tech, distance, jobs_held, weights, options[request.id][1:3]
        )
        load.add(tech, request)
        remaining = [candidate for candidate in remaining if candidate.id != request.id]
    return _finish([decisions[request.id] for request in requests], techs, 0.0)


@pytest.mark.parametrize("weights", [Weights(), Weights(w_distance=2, w_balance=6, w_unassigned=4)])
def test_incremental_regret_matches_recomputing_every_round(weights):
    for seed in range(15):
        scenario = generate(10, 30, seed=seed, scarcity=0.5)
        fast = run_regret(scenario.technicians, scenario.requests, weights)
        slow = _regret_recomputed(scenario.technicians, scenario.requests, weights)
        assert _pairs(fast) == _pairs(slow)
        assert fast.metrics.total_cost == pytest.approx(slow.metrics.total_cost)


def test_api_accepts_weights_and_rejects_unknown_distance():
    scenario = client.get("/api/scenario/trap").json()
    body = {"technicians": scenario["technicians"], "requests": scenario["requests"]}
    assert client.post("/api/allocate", json={**body, "distance": "road"}).status_code == 422
    assert client.post("/api/allocate", json={**body, "capacity": "minutes"}).status_code == 422
    unprotected = {"w_distance": 5, "w_balance": 2, "w_unassigned": 5, "must_serve_p5": False}
    response = client.post("/api/allocate", json={**body, "weights": unprotected}).json()
    assert response["results"]["hungarian"]["metrics"]["high_priority_unassigned_count"] == 1
