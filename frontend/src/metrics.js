export const ALGORITHMS = [
  { id: "greedy", label: "Greedy" },
  { id: "hungarian", label: "Hungarian" },
  { id: "regret", label: "Regret" },
];

export const METRIC_COLUMNS = [
  {
    id: "total_cost",
    label: "Total cost",
    better: "low",
    tip: "The comparison score. Green is the lowest. Each served job costs distance weight times kilometers, plus balance weight times how many jobs that technician already holds. Each dropped job costs unassigned weight times its priority.",
  },
  {
    id: "assigned",
    label: "Assigned",
    better: "high",
    tip: "Requests a technician took. Skill-gap requests are left out of this count.",
  },
  {
    id: "unassigned",
    label: "Unassigned",
    better: "low",
    tip: "Requests the algorithm left open. A skill nobody has is listed separately and is not counted here.",
  },
  {
    id: "coverage",
    label: "Coverage",
    better: "high",
    tip: "Assigned requests divided by requests the workforce could possibly serve.",
  },
  {
    id: "total_km",
    label: "Kilometers",
    better: "low",
    tip: "Travel for served jobs, using the selected distance source, at 40 km/h when checking the window. A dropped job adds no distance. This does not pick the winner.",
  },
  {
    id: "avg_km",
    label: "Avg km",
    better: "low",
    tip: "Total kilometers divided by served jobs. Blank when nothing was served.",
  },
  {
    id: "urgent_km",
    label: "Urgent km",
    better: "low",
    tip: "Travel only for priority 4 and 5 jobs that someone took. A missed urgent job adds nothing, so a smaller number can belong to the worse plan.",
  },
  {
    id: "urgent_avg_km",
    label: "Urgent avg km",
    better: "low",
    tip: "Urgent kilometers divided by served priority 4 and 5 jobs. Blank when none of those jobs were served.",
  },
  {
    id: "urgent_misses",
    label: "Urgent misses",
    better: "low",
    tip: "Priority 4 and 5 jobs left unassigned. Dropping one costs unassigned weight times that priority.",
  },
  {
    id: "workload",
    label: "Workload spread",
    better: "low",
    tip: "Standard deviation of jobs per technician. Lower means the work is shared more evenly. Not the score.",
  },
  {
    id: "runtime_ms",
    label: "Runtime ms",
    better: "low",
    tip: "How long this algorithm took. Speed is not part of the score.",
  },
  {
    id: "gap",
    label: "Gap vs Hungarian",
    better: "low",
    tip: "This total cost minus Hungarian's total cost. Zero means it matched the minimum. Hungarian is zero because it minimizes that cost.",
  },
];

export function money(value) {
  return Number(value).toFixed(1);
}

export function metricValue(metrics, id) {
  if (id === "total_cost") return metrics.total_cost;
  if (id === "assigned") return metrics.assigned_count;
  if (id === "unassigned") return metrics.unassigned_count;
  if (id === "coverage") return metrics.priority_coverage;
  if (id === "total_km") return metrics.total_km;
  if (id === "avg_km") return metrics.avg_km;
  if (id === "urgent_km") return metrics.urgent_assigned_km;
  if (id === "urgent_avg_km") return metrics.urgent_avg_km;
  if (id === "urgent_misses") return metrics.high_priority_unassigned_count;
  if (id === "workload") return metrics.workload_std;
  if (id === "runtime_ms") return metrics.runtime_ms;
  return metrics.optimality_gap;
}

export function metricText(metrics, id) {
  const value = metricValue(metrics, id);
  if (value == null) return "—";
  if (id === "assigned" || id === "unassigned" || id === "urgent_misses") return String(value);
  if (id === "coverage") return `${(value * 100).toFixed(0)}%`;
  return money(value);
}

export function skillLabel(skill) {
  if (skill === "hvac") return "HVAC";
  return skill.charAt(0).toUpperCase() + skill.slice(1);
}

export function bestValues(results) {
  const marks = {};
  if (!results) return marks;
  for (const column of METRIC_COLUMNS) {
    const values = ALGORITHMS.map((algo) => metricValue(results[algo.id].metrics, column.id)).filter((value) => value != null);
    if (!values.length) continue;
    marks[column.id] = column.better === "high" ? Math.max(...values) : Math.min(...values);
  }
  return marks;
}
