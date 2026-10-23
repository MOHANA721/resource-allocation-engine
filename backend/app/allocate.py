"""Greedy, Hungarian, and regret on one cost function and one feasibility rule."""

import math
import time

import numpy as np
from scipy.optimize import linear_sum_assignment

from app.constraints import assignment_cost, check, pair_distance, unserved_penalty
from app.models import (
    AlgorithmResult,
    AllocateResponse,
    AllocateResults,
    Assignment,
    Metrics,
    Request,
    SkillGap,
    Technician,
    Weights,
)


def skill_gaps(techs: list[Technician], requests: list[Request]) -> tuple[list[Request], list[SkillGap]]:
    offered = {skill for tech in techs for skill in tech.skills}
    live: list[Request] = []
    gaps: list[SkillGap] = []
    for request in requests:
        if request.required_skill in offered:
            live.append(request)
            continue
        gaps.append(
            SkillGap(
                request_id=request.id,
                customer=request.customer,
                required_skill=request.required_skill,
                reason=f"Skill gap: nobody on this workforce has {request.required_skill}.",
            )
        )
    return live, gaps


def _metrics(assignments: list[Assignment], techs: list[Technician], started: float, clock) -> Metrics:
    served = [row for row in assignments if row.technician_id is not None]
    missed = [row for row in assignments if row.technician_id is None]
    distances = [row.distance_km or 0.0 for row in served]
    urgent = [row for row in served if row.priority >= 4]
    urgent_km = sum(row.distance_km or 0.0 for row in urgent)
    load = [sum(1 for row in served if row.technician_id == tech.id) for tech in techs]
    mean_load = sum(load) / len(load) if load else 0.0
    variance = sum((value - mean_load) ** 2 for value in load) / len(load) if load else 0.0
    considered = len(assignments)
    return Metrics(
        total_cost=sum(row.cost for row in assignments),
        total_km=sum(distances),
        avg_km=(sum(distances) / len(distances)) if distances else None,
        assigned_count=len(served),
        unassigned_count=len(missed),
        priority_coverage=(len(served) / considered) if considered else 1.0,
        high_priority_unassigned_count=sum(1 for row in missed if row.priority >= 4),
        high_priority_avg_km=(urgent_km / len(urgent)) if urgent else None,
        urgent_assigned_km=urgent_km,
        urgent_avg_km=(urgent_km / len(urgent)) if urgent else None,
        workload_std=math.sqrt(variance),
        runtime_ms=(clock() - started) * 1000.0,
    )


def _finish(assignments: list[Assignment], techs: list[Technician], started: float) -> AlgorithmResult:
    return AlgorithmResult(assignments=assignments, metrics=_metrics(assignments, techs, started, time.perf_counter))


def _drop_reason(request: Request, techs: list[Technician], held: dict[str, int], weights: Weights) -> str:
    skilled = [tech for tech in techs if request.required_skill in tech.skills]
    if not skilled:
        return f"Unserved {request.customer}: no technician has {request.required_skill}."
    if skilled and all(held.get(tech.id, 0) >= tech.max_jobs for tech in skilled):
        return f"Unserved {request.customer}: every technician with {request.required_skill} is already full."
    if any(check(tech, request, held.get(tech.id, 0))[0] for tech in skilled):
        penalty = unserved_penalty(request.priority, weights)
        return (
            f"Unserved {request.customer}: left open because serving it cost more "
            f"than the drop penalty of {penalty:.1f}."
        )
    return (
        f"Unserved {request.customer}: no technician with {request.required_skill} "
        "can arrive before the window closes and finish inside his shift."
    )


def _served_sentence(
    request: Request,
    chosen: Technician,
    distance_km: float,
    jobs_held: int,
    weights: Weights,
    alternatives: list[tuple[float, str, float]],
) -> str:
    cost = assignment_cost(distance_km, jobs_held, weights)
    if alternatives:
        runner_cost, runner_name, runner_km = alternatives[0]
        runner_text = f" Runner-up {runner_name} at {runner_km:.1f} km (cost {runner_cost:.1f})."
    else:
        runner_text = " No other feasible technician."
    held_word = "job" if jobs_held == 1 else "jobs"
    return (
        f"Priority P{request.priority}. {chosen.name} was the cheapest of "
        f"{len(alternatives) + 1} feasible technicians at {distance_km:.1f} km "
        f"with {jobs_held} {held_word} already held (cost {cost:.1f}).{runner_text}"
    )


def _options(request: Request, techs: list[Technician], held: dict[str, int], weights: Weights):
    options = []
    for tech in techs:
        jobs_held = held.get(tech.id, 0)
        if not check(tech, request, jobs_held)[0]:
            continue
        distance = pair_distance(tech, request)
        cost = assignment_cost(distance, jobs_held, weights)
        options.append((cost, tech.id, tech, distance, jobs_held))
    options.sort(key=lambda item: (item[0], item[1]))
    return options


def _assignment_from_option(request, tech, distance, jobs_held, weights, runners) -> Assignment:
    alternatives = [(cost, runner.name, runner_distance) for cost, _id, runner, runner_distance, _held in runners]
    return Assignment(
        request_id=request.id,
        customer=request.customer,
        technician_id=tech.id,
        technician_name=tech.name,
        distance_km=distance,
        cost=assignment_cost(distance, jobs_held, weights),
        explanation=_served_sentence(request, tech, distance, jobs_held, weights, alternatives),
        priority=request.priority,
        required_skill=request.required_skill,
    )


def _unassigned(request, techs, held, weights) -> Assignment:
    return Assignment(
        request_id=request.id,
        customer=request.customer,
        cost=unserved_penalty(request.priority, weights),
        explanation=_drop_reason(request, techs, held, weights),
        priority=request.priority,
        required_skill=request.required_skill,
    )


def run_greedy(techs: list[Technician], requests: list[Request], weights: Weights) -> AlgorithmResult:
    started = time.perf_counter()
    order = sorted(requests, key=lambda item: (-item.priority, item.window_end, item.id))
    held = {tech.id: 0 for tech in techs}
    chosen: dict[str, Assignment] = {}
    for request in order:
        options = _options(request, techs, held, weights)
        if not options:
            chosen[request.id] = _unassigned(request, techs, held, weights)
            continue
        _cost, _tech_id, tech, distance, jobs_held = options[0]
        chosen[request.id] = _assignment_from_option(request, tech, distance, jobs_held, weights, options[1:3])
        held[tech.id] += 1
    return _finish([chosen[request.id] for request in requests], techs, started)


def run_regret(techs: list[Technician], requests: list[Request], weights: Weights) -> AlgorithmResult:
    started = time.perf_counter()
    remaining = list(requests)
    held = {tech.id: 0 for tech in techs}
    decisions: dict[str, Assignment] = {}
    while remaining:
        forced: list[Request] = []
        scored = []
        for request in remaining:
            options = _options(request, techs, held, weights)
            if not options:
                forced.append(request)
                continue
            gap = math.inf if len(options) == 1 else options[1][0] - options[0][0]
            scored.append((gap, request, options[0], options[1:3]))
        if forced:
            for request in forced:
                decisions[request.id] = _unassigned(request, techs, held, weights)
            forced_ids = {request.id for request in forced}
            remaining = [request for request in remaining if request.id not in forced_ids]
            continue
        scored.sort(key=lambda item: (-item[0], -item[1].priority, item[1].window_end, item[1].id))
        _gap, request, best, runners = scored[0]
        _cost, _tech_id, tech, distance, jobs_held = best
        row = _assignment_from_option(request, tech, distance, jobs_held, weights, runners)
        row.explanation += " Chosen early because this request had the largest regret."
        decisions[request.id] = row
        held[tech.id] += 1
        remaining = [candidate for candidate in remaining if candidate.id != request.id]
    return _finish([decisions[request.id] for request in requests], techs, started)


def run_hungarian(techs: list[Technician], requests: list[Request], weights: Weights) -> AlgorithmResult:
    started = time.perf_counter()
    if not requests:
        return _finish([], techs, started)
    slots = [(tech, slot) for tech in techs for slot in range(tech.max_jobs)]
    row_count = len(requests)
    slot_count = len(slots)
    sample_costs = [unserved_penalty(request.priority, weights) for request in requests]
    for request in requests:
        for tech, slot in slots:
            if check(tech, request, slot)[0]:
                sample_costs.append(assignment_cost(pair_distance(tech, request), slot, weights))
    blocker = max(sample_costs) + 1000.0
    matrix = np.full((row_count, slot_count + row_count), blocker, dtype=float)
    for row, request in enumerate(requests):
        matrix[row, slot_count + row] = unserved_penalty(request.priority, weights)
        for col, (tech, slot) in enumerate(slots):
            if check(tech, request, slot)[0]:
                matrix[row, col] = assignment_cost(pair_distance(tech, request), slot, weights)

    chosen_cols = {int(row): int(col) for row, col in zip(*linear_sum_assignment(matrix))}
    pending: list[tuple[Request, Assignment | None]] = []
    used: dict[str, int] = {}
    for row, request in enumerate(requests):
        col = chosen_cols[row]
        if col >= slot_count or float(matrix[row, col]) >= blocker - 1e-6:
            pending.append((request, None))
            continue
        tech, slot = slots[col]
        if not check(tech, request, slot)[0] or used.get(tech.id, 0) >= tech.max_jobs:
            raise RuntimeError(f"Hungarian selected an infeasible slot for {request.id}")
        used[tech.id] = used.get(tech.id, 0) + 1
        distance = pair_distance(tech, request)
        runners = []
        for other_col, (other, other_slot) in enumerate(slots):
            if other_col == col or not check(other, request, other_slot)[0]:
                continue
            if other.id == tech.id:
                continue
            runners.append((float(matrix[row, other_col]), other.id, other, pair_distance(other, request), other_slot))
        runners.sort(key=lambda item: (item[0], item[1]))
        held_word = "job" if slot == 1 else "jobs"
        closer = [
            (other_cost, other.name, other_distance)
            for other_cost, _other_id, other, other_distance, _other_slot in runners
            if other_cost + 1e-6 < float(matrix[row, col])
        ]
        if closer:
            other_cost, other_name, other_distance = closer[0]
            tradeoff = (
                f" {other_name} is closer at {other_distance:.1f} km (cost {other_cost:.1f}) "
                "and is kept for another request."
            )
        elif runners:
            other_cost, _other_id, other, other_distance, _other_slot = runners[0]
            tradeoff = f" Runner-up {other.name} at {other_distance:.1f} km (cost {other_cost:.1f})."
        else:
            tradeoff = " No other feasible technician."
        row_assignment = Assignment(
            request_id=request.id,
            customer=request.customer,
            technician_id=tech.id,
            technician_name=tech.name,
            distance_km=distance,
            cost=float(matrix[row, col]),
            explanation=(
                f"Priority P{request.priority}. {tech.name} takes this job in the minimum-cost matching "
                f"at {distance:.1f} km with {slot} {held_word} already held "
                f"(cost {float(matrix[row, col]):.1f}).{tradeoff}"
            ),
            priority=request.priority,
            required_skill=request.required_skill,
        )
        pending.append((request, row_assignment))
    assignments = [
        row if row is not None else _hungarian_unassigned(request, techs, weights)
        for request, row in pending
    ]
    return _finish(assignments, techs, started)


def _hungarian_unassigned(request: Request, techs: list[Technician], weights: Weights) -> Assignment:
    """Hungarian drops a servable request only because the whole plan is cheaper without it."""
    options = _options(request, techs, {}, weights)
    if not options:
        return _unassigned(request, techs, {}, weights)
    _cost, _tech_id, nearest, distance, _held = options[0]
    penalty = unserved_penalty(request.priority, weights)
    return Assignment(
        request_id=request.id,
        customer=request.customer,
        cost=penalty,
        explanation=(
            f"Unserved {request.customer}: left open in the minimum-cost plan. "
            f"Leaving it open costs {penalty:.1f}, which is less than rearranging "
            f"technicians to serve it. Nearest qualified technician: {nearest.name} "
            f"at {distance:.1f} km."
        ),
        priority=request.priority,
        required_skill=request.required_skill,
    )


def allocate(techs: list[Technician], requests: list[Request], weights: Weights | None = None) -> AllocateResponse:
    weights = weights or Weights()
    live, gaps = skill_gaps(techs, requests)
    greedy = run_greedy(techs, live, weights)
    hungarian = run_hungarian(techs, live, weights)
    regret = run_regret(techs, live, weights)
    optimal = hungarian.metrics.total_cost
    for result in (greedy, hungarian, regret):
        result.metrics.optimality_gap = result.metrics.total_cost - optimal
    return AllocateResponse(
        skill_gaps=gaps,
        results=AllocateResults(greedy=greedy, hungarian=hungarian, regret=regret),
    )
