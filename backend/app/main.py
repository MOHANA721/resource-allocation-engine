"""FastAPI routes for generated scenarios and side-by-side assignment."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.allocate import allocate
from app.generator import generate, trap_scenario
from app.models import AllocateRequest, AllocateResponse, GenerateRequest, Scenario

app = FastAPI(
    title="Resource Allocation Engine",
    version="2.0.0",
    summary="Assign Bengaluru field technicians to repair requests.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/scenario", response_model=Scenario)
def create_scenario(body: GenerateRequest) -> Scenario:
    return generate(body.n_techs, body.n_requests, body.seed, body.scarcity)


@app.get("/api/scenario/trap", response_model=Scenario)
def trap() -> Scenario:
    return trap_scenario()


@app.post("/api/allocate", response_model=AllocateResponse)
def allocate_route(body: AllocateRequest) -> AllocateResponse:
    return allocate(body.technicians, body.requests, body.weights, body.capacity, body.distance)
