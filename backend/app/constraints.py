"""Distance, travel time, hard checks, and the shared cost."""

import math

from app.distances import haversine_km, km, point
from app.models import Request, Technician, Weights

__all__ = ["haversine_km"]

SPEED_KMH = 40.0


def minutes(value: str) -> int:
    hours, mins = value.split(":")
    return int(hours) * 60 + int(mins)


def travel_minutes(distance_km: float) -> float:
    return distance_km / SPEED_KMH * 60.0


def pair_distance(tech: Technician, request: Request) -> float:
    """Kilometres from the technician's home to the job, from the current distance table."""
    return km(point(tech.lat, tech.lng), point(request.lat, request.lng))


def check(tech: Technician, request: Request, jobs_held: int) -> tuple[bool, str]:
    """Feasibility from home. ``jobs_held`` is how many jobs this person already has.

    Travel starts when the shift starts. The technician must arrive before the
    window closes. Work then starts at the later of arrival and window open,
    and must finish before the shift ends. Jobs are not chained stop to stop.
    """
    if jobs_held >= tech.max_jobs:
        return False, "full"
    if request.required_skill not in tech.skills:
        return False, "skill"
    arrival = minutes(tech.shift_start) + travel_minutes(pair_distance(tech, request))
    if arrival > minutes(request.window_end):
        return False, "time"
    start = max(arrival, minutes(request.window_start))
    if start + request.duration > minutes(tech.shift_end):
        return False, "shift"
    return True, "ok"


def assignment_cost(distance_km: float, jobs_held: int, weights: Weights) -> float:
    return weights.w_distance * distance_km + weights.w_balance * jobs_held


def unserved_penalty(priority: int, weights: Weights) -> float:
    return weights.w_unassigned * priority


def protected(priority: int, weights: Weights) -> bool:
    return weights.must_serve_p5 and priority == 5


def drop_threshold(priority: int, weights: Weights) -> float:
    """Greedy and regret leave a request open only when every option costs more than this."""
    return math.inf if protected(priority, weights) else unserved_penalty(priority, weights)
