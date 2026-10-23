"""Sweeps used by REPORT.html. Run from the project root:

    .venv\\Scripts\\python.exe backend\\experiments.py
"""

from app.allocate import allocate
from app.generator import generate, trap_scenario


def _gap(greedy_cost: float, optimal: float) -> float:
    if optimal <= 1e-9:
        return 0.0
    return (greedy_cost - optimal) / optimal * 100.0


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def sweep_scarcity(seeds: int = 30) -> None:
    print("scarcity sweep: 10 technicians, 20 requests")
    for scarcity in (0.1, 0.4, 0.8, 0.95):
        gaps = []
        regret_gaps = []
        greedy_urgent = []
        hungarian_urgent = []
        for seed in range(seeds):
            scenario = generate(10, 20, seed, scarcity)
            outcome = allocate(scenario.technicians, scenario.requests)
            greedy = outcome.results.greedy.metrics
            hungarian = outcome.results.hungarian.metrics
            regret = outcome.results.regret.metrics
            gaps.append(_gap(greedy.total_cost, hungarian.total_cost))
            regret_gaps.append(_gap(regret.total_cost, hungarian.total_cost))
            greedy_urgent.append(greedy.high_priority_avg_km or 0.0)
            hungarian_urgent.append(hungarian.high_priority_avg_km or 0.0)
        print(
            f"  scarcity {scarcity:.2f}  greedy gap {_mean(gaps):.2f}%  "
            f"regret gap {_mean(regret_gaps):.2f}%  "
            f"urgent avg km greedy {_mean(greedy_urgent):.2f} hungarian {_mean(hungarian_urgent):.2f}"
        )


def sweep_load(seeds: int = 30) -> None:
    print("load sweep: 10 technicians, scarcity 0.4")
    for n_requests in (8, 16, 24, 32):
        gaps = []
        regret_gaps = []
        runtimes = []
        for seed in range(seeds):
            scenario = generate(10, n_requests, seed, 0.4)
            outcome = allocate(scenario.technicians, scenario.requests)
            greedy = outcome.results.greedy.metrics
            hungarian = outcome.results.hungarian.metrics
            regret = outcome.results.regret.metrics
            gaps.append(_gap(greedy.total_cost, hungarian.total_cost))
            regret_gaps.append(_gap(regret.total_cost, hungarian.total_cost))
            runtimes.append((greedy.runtime_ms, hungarian.runtime_ms, regret.runtime_ms))
        print(
            f"  requests {n_requests:2d}  greedy gap {_mean(gaps):.2f}%  "
            f"regret gap {_mean(regret_gaps):.2f}%  "
            f"runtime ms greedy {_mean([item[0] for item in runtimes]):.1f} "
            f"hungarian {_mean([item[1] for item in runtimes]):.1f} "
            f"regret {_mean([item[2] for item in runtimes]):.1f}"
        )


def sweep_minutes(seeds: int = 30) -> None:
    print("working-minutes capacity: 10 technicians, scarcity 0.4")
    for n_requests in (16, 32):
        gaps = {"greedy": [], "regret": []}
        beaten = 0
        for seed in range(seeds):
            scenario = generate(10, n_requests, seed, 0.4)
            outcome = allocate(scenario.technicians, scenario.requests, capacity="minutes")
            optimal = outcome.results.hungarian.metrics.total_cost
            for name in gaps:
                cost = getattr(outcome.results, name).metrics.total_cost
                gaps[name].append(_gap(cost, optimal))
                beaten += cost < optimal - 1e-6
        print(
            f"  requests {n_requests:2d}  greedy gap {_mean(gaps['greedy']):.2f}%  "
            f"regret gap {_mean(gaps['regret']):.2f}%  "
            f"heuristic cheaper than Hungarian in {beaten} of {2 * seeds}"
        )


def show_trap() -> None:
    scenario = trap_scenario()
    outcome = allocate(scenario.technicians, scenario.requests)
    print("trap scenario")
    for name in ("greedy", "hungarian", "regret"):
        result = getattr(outcome.results, name)
        pairs = {row.request_id: row.technician_name or "unassigned" for row in result.assignments}
        print(
            f"  {name:9s} cost {result.metrics.total_cost:.2f}  "
            f"gap {result.metrics.optimality_gap:.2f}  "
            f"urgent misses {result.metrics.high_priority_unassigned_count}  {pairs}"
        )


if __name__ == "__main__":
    show_trap()
    sweep_scarcity()
    sweep_load()
    sweep_minutes()
