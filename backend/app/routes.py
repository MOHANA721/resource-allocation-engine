"""Ordered daily routes: a technician drives home, job, job, ... and every stop must fit."""

from app.constraints import minutes, travel_minutes
from app.distances import km, point
from app.models import Request, Technician


def route_km(tech: Technician, stops: list[Request]) -> float | None:
    """Kilometres driven on this route, or None if any stop misses its window or the shift.

    The technician leaves home at shift start. At each stop he must arrive
    before the window closes, waits for the window to open if early, and
    leaves when the job is done. The last job must finish inside his shift.
    """
    clock = float(minutes(tech.shift_start))
    shift_end = minutes(tech.shift_end)
    here = point(tech.lat, tech.lng)
    total = 0.0
    for stop in stops:
        there = point(stop.lat, stop.lng)
        leg = km(here, there)
        arrival = clock + travel_minutes(leg)
        if arrival > minutes(stop.window_end):
            return None
        clock = max(arrival, minutes(stop.window_start)) + stop.duration
        if clock > shift_end:
            return None
        total += leg
        here = there
    return total


def best_insertion(
    tech: Technician,
    stops: list[Request],
    request: Request,
    length: float,
) -> tuple[float, int] | None:
    """Fewest added kilometres over every feasible position, as (added_km, position)."""
    best: tuple[float, int] | None = None
    for position in range(len(stops) + 1):
        trial = route_km(tech, [*stops[:position], request, *stops[position:]])
        if trial is None:
            continue
        added = trial - length
        if best is None or added < best[0] - 1e-12:
            best = (added, position)
    return best


def build_route(tech: Technician, requests: list[Request]):
    """Insert jobs one at a time, earliest window close first.

    Returns (stops, added km per request id, order index per request id), or
    the first request that does not fit.
    """
    stops: list[Request] = []
    length = 0.0
    added: dict[str, float] = {}
    order: dict[str, int] = {}
    for index, request in enumerate(sorted(requests, key=lambda item: (item.window_end, item.id))):
        insertion = best_insertion(tech, stops, request, length)
        if insertion is None:
            return request
        extra, position = insertion
        stops.insert(position, request)
        length += extra
        added[request.id] = extra
        order[request.id] = index
    return stops, added, order
