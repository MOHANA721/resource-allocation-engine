"""Where kilometres come from: OSRM road distances, or straight line × 1.35.

``allocate`` builds one table per call and makes it current for the duration
of that call; every distance lookup in the engine goes through ``km``.
"""

import contextvars
import itertools
import json
import math
import os
import threading
import time
import urllib.error
import urllib.request
from contextlib import contextmanager

from app.models import DistanceSource

Point = tuple[float, float]

DETOUR_FACTOR = 1.35
OSRM_URL = os.environ.get("RAE_OSRM_URL", "https://router.project-osrm.org").rstrip("/")
OSRM_MAX_COORDINATES = 100
OSRM_MIN_INTERVAL_S = 1.0
OSRM_TIMEOUT_S = 15.0
CACHE_LIMIT = 200_000


class OsrmError(RuntimeError):
    pass


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


def point(lat: float, lng: float) -> Point:
    return round(lat, 6), round(lng, 6)


class DistanceTable:
    def __init__(self, source: str, note: str, factor: float = 1.0, pairs: dict | None = None):
        self.source = source
        self.note = note
        self.factor = factor
        self.pairs = pairs or {}

    def km(self, a: Point, b: Point) -> float:
        if a == b:
            return 0.0
        road = self.pairs.get((a, b))
        if road is not None:
            return road
        return haversine_km(*a, *b) * self.factor


_DETOUR_NOTE = f"Straight-line distances × {DETOUR_FACTOR} for road detours."
_DETOUR = DistanceTable("detour", _DETOUR_NOTE, factor=DETOUR_FACTOR)
_current: contextvars.ContextVar[DistanceTable] = contextvars.ContextVar("distance_table", default=_DETOUR)
_cache: dict[tuple[Point, Point], float | None] = {}
_osrm_lock = threading.Lock()
_last_call = 0.0


def km(a: Point, b: Point) -> float:
    return _current.get().km(a, b)


@contextmanager
def using(table: DistanceTable):
    token = _current.set(table)
    try:
        yield table
    finally:
        _current.reset(token)


def _fetch(points: list[Point]) -> list[list[float | None]]:
    """All-to-all road distances in km for up to OSRM_MAX_COORDINATES points."""
    global _last_call
    coords = ";".join(f"{lng:.6f},{lat:.6f}" for lat, lng in points)
    url = f"{OSRM_URL}/table/v1/driving/{coords}?annotations=distance"
    request = urllib.request.Request(url, headers={"User-Agent": "resource-allocation-engine"})
    with _osrm_lock:
        wait = OSRM_MIN_INTERVAL_S - (time.monotonic() - _last_call)
        if wait > 0:
            time.sleep(wait)
        try:
            with urllib.request.urlopen(request, timeout=OSRM_TIMEOUT_S) as response:
                body = json.load(response)
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as error:
            raise OsrmError(str(getattr(error, "reason", error))) from error
        finally:
            _last_call = time.monotonic()
    if body.get("code") != "Ok":
        raise OsrmError(body.get("message") or body.get("code") or "bad response")
    return [[None if metres is None else metres / 1000.0 for metres in row] for row in body["distances"]]


def _store(points: list[Point], matrix: list[list[float | None]]) -> None:
    for i, a in enumerate(points):
        for j, b in enumerate(points):
            if a != b:
                _cache[(a, b)] = matrix[i][j]


def road_pairs(points: list[Point]) -> dict[tuple[Point, Point], float]:
    """Road km between every two distinct points, from the cache or OSRM.

    The public server takes at most 100 coordinates per call, so larger sets
    are split into blocks of 50 and every pair of blocks is fetched together.
    """
    unique = sorted(set(points))
    missing = any((a, b) not in _cache for a in unique for b in unique if a != b)
    if missing:
        if len(_cache) > CACHE_LIMIT:
            _cache.clear()
        if len(unique) <= OSRM_MAX_COORDINATES:
            _store(unique, _fetch(unique))
        else:
            size = OSRM_MAX_COORDINATES // 2
            blocks = [unique[start:start + size] for start in range(0, len(unique), size)]
            for first, second in itertools.combinations(blocks, 2):
                group = first + second
                _store(group, _fetch(group))
    return {
        (a, b): _cache[(a, b)]
        for a in unique
        for b in unique
        if a != b and _cache.get((a, b)) is not None
    }


def build(points: list[Point], source: DistanceSource) -> DistanceTable:
    if source == "detour":
        return _DETOUR
    try:
        pairs = road_pairs(points)
    except OsrmError as error:
        return DistanceTable(
            "detour",
            f"OSRM road distances were unavailable ({error}). Using straight-line distances × {DETOUR_FACTOR}.",
            factor=DETOUR_FACTOR,
        )
    return DistanceTable("osrm", "Road distances from OSRM (OpenStreetMap).", factor=DETOUR_FACTOR, pairs=pairs)
