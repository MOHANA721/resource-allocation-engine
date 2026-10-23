import { describe, expect, it } from "vitest";
import { ALGORITHMS, METRIC_COLUMNS, bestValues, metricText, metricValue, skillLabel } from "./metrics";

function metrics(overrides = {}) {
  return {
    total_cost: 10,
    total_km: 8,
    avg_km: 4,
    assigned_count: 2,
    unassigned_count: 0,
    priority_coverage: 1,
    high_priority_unassigned_count: 0,
    high_priority_avg_km: 4,
    urgent_assigned_km: 8,
    urgent_avg_km: 4,
    workload_std: 0,
    runtime_ms: 1.25,
    optimality_gap: 0,
    ...overrides,
  };
}

const trap = {
  greedy: { metrics: metrics({ total_cost: 150.26, assigned_count: 1, unassigned_count: 1, priority_coverage: 0.5, high_priority_unassigned_count: 1, urgent_assigned_km: 0.26, total_km: 0.26, workload_std: 0.5, optimality_gap: 143.99 }) },
  hungarian: { metrics: metrics({ total_cost: 6.27, total_km: 6.27, urgent_assigned_km: 6.27 }) },
  regret: { metrics: metrics({ total_cost: 6.27, total_km: 6.27, urgent_assigned_km: 6.27, runtime_ms: 0.1 }) },
};

describe("metric columns", () => {
  it("every column has a label, a direction, and a hover tip", () => {
    for (const column of METRIC_COLUMNS) {
      expect(column.label).toBeTruthy();
      expect(["low", "high"]).toContain(column.better);
      expect(column.tip.length).toBeGreaterThan(20);
    }
  });

  it("every column reads a field the API returns", () => {
    const sample = metrics();
    for (const column of METRIC_COLUMNS) {
      expect(metricValue(sample, column.id)).not.toBeUndefined();
    }
  });

  it("lists the three algorithms in display order", () => {
    expect(ALGORITHMS.map((algo) => algo.id)).toEqual(["greedy", "hungarian", "regret"]);
  });
});

describe("metricText", () => {
  it("formats counts, percentages, and decimals", () => {
    const sample = metrics({ priority_coverage: 0.95, total_cost: 166.666 });
    expect(metricText(sample, "assigned")).toBe("2");
    expect(metricText(sample, "coverage")).toBe("95%");
    expect(metricText(sample, "total_cost")).toBe("166.7");
  });

  it("shows a dash when an average has no served jobs", () => {
    expect(metricText(metrics({ avg_km: null, urgent_avg_km: null }), "avg_km")).toBe("—");
    expect(metricText(metrics({ urgent_avg_km: null }), "urgent_avg_km")).toBe("—");
  });
});

describe("bestValues", () => {
  it("marks the lowest cost and the highest coverage on the trap", () => {
    const best = bestValues(trap);
    expect(best.total_cost).toBe(6.27);
    expect(best.assigned).toBe(2);
    expect(best.coverage).toBe(1);
    expect(best.urgent_misses).toBe(0);
  });

  it("marks greedy's short urgent travel as lowest even though it loses on cost", () => {
    expect(bestValues(trap).urgent_km).toBe(0.26);
  });

  it("returns nothing before results arrive", () => {
    expect(bestValues(undefined)).toEqual({});
  });
});

describe("skillLabel", () => {
  it("writes HVAC in capitals and capitalizes the other skills", () => {
    expect(skillLabel("hvac")).toBe("HVAC");
    expect(skillLabel("plumbing")).toBe("Plumbing");
  });
});
