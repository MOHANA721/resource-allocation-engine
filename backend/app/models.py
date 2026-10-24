"""Request and response models for multi-job field service."""

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


SKILLS = ("hvac", "electrical", "plumbing", "carpentry", "appliance")


class Technician(BaseModel):
    id: str
    name: str
    lat: float
    lng: float
    area: str
    skills: list[str] = Field(min_length=1)
    shift_start: str
    shift_end: str
    max_jobs: int = Field(ge=1)


class Request(BaseModel):
    id: str
    customer: str
    lat: float
    lng: float
    area: str
    required_skill: str
    window_start: str
    window_end: str
    duration: int = Field(ge=1, description="Job length in minutes")
    priority: int = Field(ge=1, le=5)


class Weights(BaseModel):
    w_distance: float = Field(1.0, ge=0)
    w_balance: float = Field(2.0, ge=0)
    w_unassigned: float = Field(30.0, ge=0)
    must_serve_p5: bool = Field(True, description="Never leave a priority-5 request open if it can be served")


DistanceSource = Literal["osrm", "detour"]


class Assignment(BaseModel):
    request_id: str
    customer: str
    technician_id: Optional[str] = None
    technician_name: Optional[str] = None
    distance_km: Optional[float] = None
    cost: float
    explanation: str
    priority: int
    required_skill: str


class Metrics(BaseModel):
    total_cost: float
    total_km: float
    avg_km: Optional[float] = None
    assigned_count: int
    unassigned_count: int
    priority_coverage: float
    high_priority_unassigned_count: int
    high_priority_avg_km: Optional[float] = None
    urgent_assigned_km: float = 0.0
    urgent_avg_km: Optional[float] = None
    workload_std: float
    runtime_ms: float
    optimality_gap: float = 0.0


class AlgorithmResult(BaseModel):
    assignments: list[Assignment]
    metrics: Metrics


class Scenario(BaseModel):
    label: str
    seed: Optional[int] = None
    technicians: list[Technician]
    requests: list[Request]


class GenerateRequest(BaseModel):
    n_techs: int = Field(12, ge=1, le=40)
    n_requests: int = Field(20, ge=0, le=80)
    seed: int = 1
    scarcity: float = Field(0.4, ge=0, le=1)


class AllocateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    technicians: list[Technician]
    requests: list[Request]
    weights: Weights = Weights()
    distance: DistanceSource = Field(
        "osrm",
        description="osrm: road distances, falling back to straight line times 1.35 if OSRM is unreachable. "
        "detour: straight line times 1.35.",
    )


class AllocateResults(BaseModel):
    greedy: AlgorithmResult
    hungarian: AlgorithmResult
    regret: AlgorithmResult


class AllocateResponse(BaseModel):
    results: AllocateResults
    distance_source: str = "detour"
    distance_note: str = "Straight-line distances × 1.35 for road detours."
