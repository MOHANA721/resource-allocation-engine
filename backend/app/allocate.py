"""Greedy, Hungarian, and regret on one cost function and one feasibility rule."""

import math
import time

import numpy as np
from scipy.optimize import linear_sum_assignment

from app.constraints import (
    assignment_cost,
    check,
    drop_threshold,
    job_minutes,
    pair_distance,
    protected,
    shift_minutes,
    unserved_penalty,
)
from app.distances import build, point, using
from app.models import (
    AlgorithmResult,
    AllocateResponse,
    AllocateResults,
    Assignment,
    Capacity,
    DistanceSource,
    Metrics,
    Request,
    SkillGap,
    Technician,
    Weights,
)
from app.routes import best_insertion, build_route


class _Load:
    """What each technician holds: jobs, and in minutes mode his working minutes,
    or in routed mode his ordered route and its length."""

    def __init__(self, techs: list[Technician], capacity: Capacity):
        self.jobs = {tech.id: 0 for tech in techs}
        self.minutes = {tech.id: 0.0 for tech in techs} if capacity == "minutes" else None
        self.routes = {tech.id: [] for tech in techs} if capacity == "routed" else None
        self.lengths = {tech.id: 0.0 for tech in techs} if capacity == "routed" else None

    def used(self, tech_id: str) -> float | None:
        return None if self.minutes is None else self.minutes[tech_id]

    def option(self, tech: Technician, request: Request, weights: Weights) -> tuple[float, float] | None:
        """(cost, km) of giving this request to this technician now, or None if he cannot take it.

        In routed mode the km are what the job adds to his route at its best position.
        """
        if not check(tech, request, self.jobs[tech.id], self.used(tech.id))[0]:
            return None
        if self.routes is None:
            distance = pair_distance(tech, request)
        else:
            insertion = best_insertion(tech, self.routes[tech.id], request, self.lengths[tech.id])
            if insertion is None:
                return None
            distance = insertion[0]
        return assignment_cost(distance, self.jobs[tech.id], weights), distance

    def add(self, tech: Technician, request: Request) -> None:
        if self.routes is not None:
            added, position = best_insertion(tech, self.routes[tech.id], request, self.lengths[tech.id])
            self.routes[tech.id].insert(position, request)
            self.lengths[tech.id] += added
        self.jobs[tech.id] += 1
        if self.minutes is not None:
            self.minutes[tech.id] += job_minutes(tech, request)

    def stops(self) -> dict[str, int]:
        if self.routes is None:
            return {}
        return {request.id: index + 1 for route in self.routes.values() for index, request in enumerate(route)}


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


def _drop_reason(request: Request, techs: list[Technician], load: _Load) -> str:
    skilled = [tech for tech in techs if request.required_skill in tech.skills]
    if not skilled:
        return f"Unserved {request.customer}: no technician has {request.required_skill}."
    if all(load.jobs[tech.id] >= tech.max_jobs for tech in skilled):
        return f"Unserved {request.customer}: every technician with {request.required_skill} is already full."
    reachable = any(check(tech, request, load.jobs[tech.id])[0] for tech in skilled)
    if load.minutes is not None and reachable:
        return (
            f"Unserved {request.customer}: every technician with {request.required_skill} "
            "who could reach it has run out of working time today."
        )
    if load.routes is not None and reachable:
        return (
            f"Unserved {request.customer}: no technician with {request.required_skill} can fit it "
            "into his route before its window closes and inside his shift."
        )
    return (
        f"Unserved {request.customer}: no technician with {request.required_skill} "
        "can arrive before the window closes and finish inside his shift."
    )


def _km_phrase(distance_km: float, routed: bool) -> str:
    return f"adding {distance_km:.1f} km to his route" if routed else f"at {distance_km:.1f} km"


def _served_sentence(
    request: Request,
    chosen: Technician,
    distance_km: float,
    jobs_held: int,
    weights: Weights,
    alternatives: list[tuple[float, str, float]],
    routed: bool = False,
) -> str:
    cost = assignment_cost(distance_km, jobs_held, weights)
    if alternatives:
        runner_cost, runner_name, runner_km = alternatives[0]
        runner_text = f" Runner-up {runner_name} {_km_phrase(runner_km, routed)} (cost {runner_cost:.1f})."
    else:
        runner_text = " No other feasible technician."
    held_word = "job" if jobs_held == 1 else "jobs"
    return (
        f"Priority P{request.priority}. {chosen.name} was the cheapest of "
        f"{len(alternatives) + 1} feasible technicians {_km_phrase(distance_km, routed)} "
        f"with {jobs_held} {held_word} already held (cost {cost:.1f}).{runner_text}"
    )


def _option_key(option) -> tuple[float, str]:
    return option[0], option[1]


def _options(request: Request, techs: list[Technician], load: _Load, weights: Weights):
    options = []
    for tech in techs:
        offer = load.option(tech, request, weights)
        if offer is not None:
            options.append((offer[0], tech.id, tech, offer[1], load.jobs[tech.id]))
    options.sort(key=_option_key)
    return options


def _assignment_from_option(request, tech, distance, jobs_held, weights, runners, routed=False) -> Assignment:
    alternatives = [(cost, runner.name, runner_distance) for cost, _id, runner, runner_distance, _held in runners]
    return Assignment(
        request_id=request.id,
        customer=request.customer,
        technician_id=tech.id,
        technician_name=tech.name,
        distance_km=distance,
        cost=assignment_cost(distance, jobs_held, weights),
        explanation=_served_sentence(request, tech, distance, jobs_held, weights, alternatives, routed),
        priority=request.priority,
        required_skill=request.required_skill,
    )


def _number_stops(rows: dict[str, Assignment], load: _Load) -> None:
    for request_id, stop in load.stops().items():
        rows[request_id].stop = stop


def _unassigned(request: Request, techs: list[Technician], load: _Load, weights: Weights) -> Assignment:
    return Assignment(
        request_id=request.id,
        customer=request.customer,
        cost=unserved_penalty(request.priority, weights),
        explanation=_drop_reason(request, techs, load),
        priority=request.priority,
        required_skill=request.required_skill,
    )


def _declined(request: Request, cheapest, weights: Weights) -> Assignment:
    cost, _tech_id, tech, distance, _held = cheapest
    penalty = unserved_penalty(request.priority, weights)
    return Assignment(
        request_id=request.id,
        customer=request.customer,
        cost=penalty,
        explanation=(
            f"Unserved {request.customer}: left open on purpose. The cheapest option, {tech.name} "
            f"at {distance:.1f} km, costs {cost:.1f}, more than the penalty of {penalty:.1f} "
            "for leaving it open."
        ),
        priority=request.priority,
        required_skill=request.required_skill,
    )


def _open_or_decline(request, options, techs, load, weights) -> Assignment | None:
    """The unserved row for a request greedy or regret will not serve, or None if it should be served."""
    if not options:
        return _unassigned(request, techs, load, weights)
    if options[0][0] > drop_threshold(request.priority, weights):
        return _declined(request, options[0], weights)
    return None


def run_greedy(
    techs: list[Technician],
    requests: list[Request],
    weights: Weights,
    capacity: Capacity = "jobs",
) -> AlgorithmResult:
    started = time.perf_counter()
    order = sorted(requests, key=lambda item: (-item.priority, item.window_end, item.id))
    load = _Load(techs, capacity)
    chosen: dict[str, Assignment] = {}
    for request in order:
        options = _options(request, techs, load, weights)
        missed = _open_or_decline(request, options, techs, load, weights)
        if missed is not None:
            chosen[request.id] = missed
            continue
        _cost, _tech_id, tech, distance, jobs_held = options[0]
        chosen[request.id] = _assignment_from_option(
            request, tech, distance, jobs_held, weights, options[1:3], capacity == "routed"
        )
        load.add(tech, request)
    _number_stops(chosen, load)
    return _finish([chosen[request.id] for request in requests], techs, started)


def _regret_key(request: Request, options) -> tuple:
    gap = math.inf if len(options) == 1 else options[1][0] - options[0][0]
    return -request.priority, -gap, request.window_end, request.id


def run_regret(
    techs: list[Technician],
    requests: list[Request],
    weights: Weights,
    capacity: Capacity = "jobs",
) -> AlgorithmResult:
    """Highest priority first; within a priority, the request with the most to lose goes first.

    Options are built once. After each assignment only the chosen technician's
    entry is recomputed in the requests that still list him; once he drops out
    of a request's list he never returns, because his load only grows.
    """
    started = time.perf_counter()
    load = _Load(techs, capacity)
    options = {request.id: _options(request, techs, load, weights) for request in requests}
    remaining = list(requests)
    decisions: dict[str, Assignment] = {}
    while remaining:
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
        best, runners = options[request.id][0], options[request.id][1:3]
        _cost, _tech_id, tech, distance, jobs_held = best
        row = _assignment_from_option(request, tech, distance, jobs_held, weights, runners, capacity == "routed")
        row.explanation += " Chosen early: among the highest-priority requests left, it had the largest regret."
        decisions[request.id] = row
        load.add(tech, request)
        remaining = [candidate for candidate in remaining if candidate.id != request.id]
        for other in remaining:
            listed = options[other.id]
            for index, item in enumerate(listed):
                if item[1] != tech.id:
                    continue
                offer = load.option(tech, other, weights)
                if offer is None:
                    del listed[index]
                else:
                    listed[index] = (offer[0], tech.id, tech, offer[1], load.jobs[tech.id])
                    listed.sort(key=_option_key)
                break
    _number_stops(decisions, load)
    return _finish([decisions[request.id] for request in requests], techs, started)


def _hungarian_matrix(requests, slots, weights, feasible, banned) -> tuple[np.ndarray, float]:
    """Slot columns, then one 'leave open' column per request.

    A protected priority-5 request pays an extra guard on its open column that
    is larger than any whole plan, so the solver serves as many of them as it can.
    """
    row_count = len(requests)
    slot_count = len(slots)
    costs = [[None] * slot_count for _ in requests]
    worst = 0.0
    for row, request in enumerate(requests):
        row_worst = unserved_penalty(request.priority, weights)
        for col, (tech, slot) in enumerate(slots):
            if feasible[row][col] and (row, tech.id) not in banned:
                costs[row][col] = assignment_cost(pair_distance(tech, request), slot, weights)
                row_worst = max(row_worst, costs[row][col])
        worst += row_worst
    base = worst + 1.0
    guarded = [protected(request.priority, weights) for request in requests]
    blocker = base * (sum(guarded) + 1) + 1000.0
    matrix = np.full((row_count, slot_count + row_count), blocker, dtype=float)
    for row, request in enumerate(requests):
        matrix[row, slot_count + row] = unserved_penalty(request.priority, weights) + (base if guarded[row] else 0.0)
        for col, cost in enumerate(costs[row]):
            if cost is not None:
                matrix[row, col] = cost
    return matrix, blocker


def _overruns(requests, slots, chosen_cols) -> set[tuple[int, str]]:
    """For each technician past his working time, the (row, technician) pair to forbid next."""
    by_tech: dict[str, list[int]] = {}
    for row, col in chosen_cols.items():
        if col < len(slots):
            by_tech.setdefault(slots[col][0].id, []).append(row)
    techs = {tech.id: tech for tech, _slot in slots}
    banned = set()
    for tech_id, rows in by_tech.items():
        tech = techs[tech_id]
        used = {row: job_minutes(tech, requests[row]) for row in rows}
        if sum(used.values()) > shift_minutes(tech):
            longest = max(rows, key=lambda row: (used[row], requests[row].id))
            banned.add((longest, tech_id))
    return banned


def _rows_by_tech(slots, chosen_cols) -> dict[str, list[int]]:
    by_tech: dict[str, list[int]] = {}
    for row, col in sorted(chosen_cols.items()):
        if col < len(slots):
            by_tech.setdefault(slots[col][0].id, []).append(row)
    return by_tech


def _route_failures(requests, slots, chosen_cols) -> set[tuple[int, str]]:
    """For each technician whose jobs cannot be driven in order, the (row, technician) pair to forbid next."""
    techs = {tech.id: tech for tech, _slot in slots}
    rows_of = {request.id: row for row, request in enumerate(requests)}
    banned = set()
    for tech_id, rows in _rows_by_tech(slots, chosen_cols).items():
        built = build_route(techs[tech_id], [requests[row] for row in rows])
        if isinstance(built, Request):
            banned.add((rows_of[built.id], tech_id))
    return banned


def run_hungarian(
    techs: list[Technician],
    requests: list[Request],
    weights: Weights,
    capacity: Capacity = "jobs",
) -> AlgorithmResult:
    """Minimum-cost matching. In minutes mode, a technician who ends up past his
    working time loses his longest job and the matching is solved again. In
    routed mode, the matching prices jobs from home; each technician's jobs are
    then put in order, and a job that does not fit is forbidden for him before
    solving again."""
    started = time.perf_counter()
    if not requests:
        return _finish([], techs, started)
    slots = [(tech, slot) for tech in techs for slot in range(tech.max_jobs)]
    slot_count = len(slots)
    start_minutes = 0.0 if capacity == "minutes" else None
    feasible = [[check(tech, request, slot, start_minutes)[0] for tech, slot in slots] for request in requests]
    banned: set[tuple[int, str]] = set()
    while True:
        matrix, blocker = _hungarian_matrix(requests, slots, weights, feasible, banned)
        chosen_cols = {int(row): int(col) for row, col in zip(*linear_sum_assignment(matrix))}
        chosen_cols = {
            row: col for row, col in chosen_cols.items() if col < slot_count and matrix[row, col] < blocker - 1e-6
        }
        if capacity == "minutes":
            extra = _overruns(requests, slots, chosen_cols)
        elif capacity == "routed":
            extra = _route_failures(requests, slots, chosen_cols)
        else:
            extra = set()
        if not extra:
            break
        banned |= extra

    routed: dict[str, tuple[float, int, int]] = {}
    built_routes: dict[str, tuple[list[Request], float]] = {}
    if capacity == "routed":
        tech_of = {tech.id: tech for tech in techs}
        for tech_id, rows in _rows_by_tech(slots, chosen_cols).items():
            stops, added, order = build_route(tech_of[tech_id], [requests[row] for row in rows])
            built_routes[tech_id] = (stops, sum(added.values()))
            for index, stop in enumerate(stops):
                routed[stop.id] = (added[stop.id], order[stop.id], index + 1)

    pending: list[tuple[Request, Assignment | None]] = []
    used: dict[str, int] = {}
    for row, request in enumerate(requests):
        col = chosen_cols.get(row)
        if col is None:
            pending.append((request, None))
            continue
        tech, slot = slots[col]
        if not feasible[row][col] or used.get(tech.id, 0) >= tech.max_jobs:
            raise RuntimeError(f"Hungarian selected an infeasible slot for {request.id}")
        used[tech.id] = used.get(tech.id, 0) + 1
        distance = pair_distance(tech, request)
        cost = float(matrix[row, col])
        stop = None
        if request.id in routed:
            distance, slot, stop = routed[request.id]
            cost = assignment_cost(distance, slot, weights)
        runners = []
        for other_col, (other, other_slot) in enumerate(slots):
            if other_col == col or other.id == tech.id:
                continue
            if not feasible[row][other_col] or (row, other.id) in banned:
                continue
            runners.append((float(matrix[row, other_col]), other.id, other, pair_distance(other, request), other_slot))
        runners.sort(key=_option_key)
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
        held_word = "job" if slot == 1 else "jobs"
        row_assignment = Assignment(
            request_id=request.id,
            customer=request.customer,
            technician_id=tech.id,
            technician_name=tech.name,
            distance_km=distance,
            cost=cost,
            explanation=(
                f"Priority P{request.priority}. {tech.name} takes this job in the minimum-cost matching "
                f"{_km_phrase(distance, stop is not None)} with {slot} {held_word} already held "
                f"(cost {cost:.1f}).{tradeoff}"
            ),
            priority=request.priority,
            required_skill=request.required_skill,
            stop=stop,
        )
        pending.append((request, row_assignment))
    final = _Load(techs, capacity)
    by_id = {tech.id: tech for tech in techs}
    if final.routes is not None:
        for tech_id, (stops, length) in built_routes.items():
            final.routes[tech_id] = list(stops)
            final.lengths[tech_id] = length
            final.jobs[tech_id] = len(stops)
    else:
        for request, row in pending:
            if row is not None:
                final.add(by_id[row.technician_id], request)
    assignments = [
        row if row is not None else _hungarian_unassigned(request, techs, weights, capacity, final)
        for request, row in pending
    ]
    return _finish(assignments, techs, started)


def _hungarian_unassigned(
    request: Request,
    techs: list[Technician],
    weights: Weights,
    capacity: Capacity,
    final: _Load,
) -> Assignment:
    """Hungarian drops a servable request only because the whole plan is better without it."""
    empty = _Load(techs, capacity)
    options = _options(request, techs, empty, weights)
    if not options:
        return _unassigned(request, techs, empty, weights)
    _cost, _tech_id, nearest, distance, _held = options[0]
    penalty = unserved_penalty(request.priority, weights)
    if protected(request.priority, weights):
        limit = " or running a technician past his working time" if capacity == "minutes" else ""
        reason = (
            f"Unserved {request.customer}: priority 5, but no plan serves it without leaving "
            f"another priority-5 request open{limit}."
        )
    elif not _options(request, techs, final, weights):
        booked = {"minutes": " by job count or working time", "routed": " by job count or route timing"}.get(
            capacity, ""
        )
        reason = (
            f"Unserved {request.customer}: left open in the minimum-cost plan. Every qualified "
            f"technician who can reach it is fully booked{booked}, and rearranging them would not "
            f"cost less than the {penalty:.1f} for leaving it open."
        )
    else:
        reason = (
            f"Unserved {request.customer}: left open in the minimum-cost plan. "
            f"Leaving it open costs {penalty:.1f}, which is less than serving it."
        )
    return Assignment(
        request_id=request.id,
        customer=request.customer,
        cost=penalty,
        explanation=f"{reason} Nearest qualified technician: {nearest.name} at {distance:.1f} km.",
        priority=request.priority,
        required_skill=request.required_skill,
    )


def allocate(
    techs: list[Technician],
    requests: list[Request],
    weights: Weights | None = None,
    capacity: Capacity = "jobs",
    distance: DistanceSource = "straight",
) -> AllocateResponse:
    """All three algorithms on one scenario. The API defaults ``distance`` to
    OSRM; this function defaults to straight line so direct calls work offline."""
    weights = weights or Weights()
    live, gaps = skill_gaps(techs, requests)
    points = [point(tech.lat, tech.lng) for tech in techs] + [point(request.lat, request.lng) for request in live]
    table = build(points, distance)
    with using(table):
        greedy = run_greedy(techs, live, weights, capacity)
        hungarian = run_hungarian(techs, live, weights, capacity)
        regret = run_regret(techs, live, weights, capacity)
    optimal = hungarian.metrics.total_cost
    for result in (greedy, hungarian, regret):
        result.metrics.optimality_gap = result.metrics.total_cost - optimal
    return AllocateResponse(
        skill_gaps=gaps,
        results=AllocateResults(greedy=greedy, hungarian=hungarian, regret=regret),
        distance_source=table.source,
        distance_note=table.note,
    )
