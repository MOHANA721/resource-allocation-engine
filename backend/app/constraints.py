"""Distance, travel time, hard checks, and the shared cost."""

import math

from app.models import Request, Technician, Weights

SPEED_KMH = 40.0


def minutes(value: str) -> int:
    hours, mins = value.split(":")
    return int(hours) * 60 + int(mins)


def haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    radius = 6371.0088
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lng2 - lng1)
    a = (
        math.sin(d_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    )
    return 2 * radius * math.asin(math.sqrt(max(0.0, a)))


def travel_minutes(distance_km: float) -> float:
    return distance_km / SPEED_KMH * 60.0


def pair_distance(tech: Technician, request: Request) -> float:
    return haversine_km(tech.lat, tech.lng, request.lat, request.lng)


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
