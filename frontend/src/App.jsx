import { useEffect, useMemo, useState } from "react";
import { CircleMarker, MapContainer, Polyline, TileLayer, Tooltip, useMap } from "react-leaflet";
import { ALGORITHMS, METRIC_COLUMNS, bestValues, metricText, metricValue, money, skillLabel } from "./metrics";

const API = "http://127.0.0.1:8000";
const PALETTE = ["#0f766e", "#b45309", "#1d4ed8", "#be123c", "#6d28d9", "#15803d", "#c2410c", "#0369a1", "#a16207", "#4338ca"];

function FitBounds({ points }) {
  const map = useMap();
  useEffect(() => {
    if (points.length) map.fitBounds(points, { padding: [24, 24] });
  }, [map, points]);
  return null;
}

function FieldServiceMap({ scenario, result, colors, selectedId, onSelect }) {
  const byTech = useMemo(() => new Map(scenario.technicians.map((tech) => [tech.id, tech])), [scenario]);
  const byRequest = useMemo(() => new Map(scenario.requests.map((request) => [request.id, request])), [scenario]);
  const points = useMemo(
    () => [
      ...scenario.technicians.map((tech) => [tech.lat, tech.lng]),
      ...scenario.requests.map((request) => [request.lat, request.lng]),
    ],
    [scenario]
  );
  return (
    <MapContainer center={[12.97, 77.59]} zoom={11} scrollWheelZoom={false}>
      <TileLayer attribution='&copy; OpenStreetMap' url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
      <FitBounds points={points} />
      {result.assignments.map((row) => {
        if (!row.technician_id) return null;
        const tech = byTech.get(row.technician_id);
        const request = byRequest.get(row.request_id);
        if (!tech || !request) return null;
        return (
          <Polyline
            key={`${row.request_id}-line`}
            positions={[[tech.lat, tech.lng], [request.lat, request.lng]]}
            pathOptions={{ color: colors.get(tech.id), weight: 3, opacity: 0.85 }}
          />
        );
      })}
      {scenario.technicians.map((tech) => (
        <CircleMarker key={tech.id} center={[tech.lat, tech.lng]} radius={8} pathOptions={{ color: "#fff", weight: 2, fillColor: colors.get(tech.id), fillOpacity: 1 }}>
          <Tooltip>{tech.name} · {tech.skills.map(skillLabel).join(", ")} · {tech.max_jobs} jobs</Tooltip>
        </CircleMarker>
      ))}
      {scenario.requests.map((request) => {
        const row = result.assignments.find((item) => item.request_id === request.id);
        const served = Boolean(row?.technician_id);
        const gap = !row;
        return (
          <CircleMarker
            key={request.id}
            center={[request.lat, request.lng]}
            radius={4 + request.priority}
            pathOptions={{
              color: selectedId === request.id ? "#111827" : gap ? "#b45309" : served ? "#166534" : "#b91c1c",
              weight: selectedId === request.id ? 3 : 2,
              fillColor: gap ? "#f59e0b" : served ? "#86efac" : "#fecaca",
              fillOpacity: served || gap ? 0.9 : 0.15,
            }}
            eventHandlers={{ click: () => onSelect(request.id) }}
          >
            <Tooltip>
              {request.customer} · P{request.priority} · {skillLabel(request.required_skill)}
              {gap ? " · skill gap" : served ? "" : " · unassigned"}
            </Tooltip>
          </CircleMarker>
        );
      })}
    </MapContainer>
  );
}

function Behavior({ response }) {
  const greedy = response.results.greedy.metrics;
  const hungarian = response.results.hungarian.metrics;
  const regret = response.results.regret.metrics;
  const shorterUrgent =
    greedy.urgent_assigned_km + 0.05 < hungarian.urgent_assigned_km &&
    greedy.total_cost > hungarian.total_cost + 0.05;
  return (
    <div className="behavior">
      <p><strong>Who fails here.</strong> The green score is total cost. Urgent kilometers explain the trips. They do not pick the winner.</p>
      <ul>
        <li>
          Greedy cost {money(greedy.total_cost)}, urgent misses {greedy.high_priority_unassigned_count}, gap {money(greedy.optimality_gap)}.
          It locks the highest priority and earliest deadline first, then never takes that choice back.
        </li>
        <li>
          Hungarian cost {money(hungarian.total_cost)}, urgent misses {hungarian.high_priority_unassigned_count}.
          It sees every slot at once, so it can leave a nearby job for a technician who is the only person who can do another job.
        </li>
        <li>
          Regret cost {money(regret.total_cost)}, urgent misses {regret.high_priority_unassigned_count}, gap {money(regret.optimality_gap)}.
          A request with only one feasible technician goes first. That is how it avoids the trap greedy falls into.
        </li>
        {shorterUrgent && (
          <li>
            Greedy urgent travel is {money(greedy.urgent_assigned_km)} km against Hungarian {money(hungarian.urgent_assigned_km)} km,
            and greedy still has the higher total cost. The short trips are the ones that stole a scarce technician.
          </li>
        )}
        {response.skill_gaps.length > 0 && (
          <li>
            {response.skill_gaps.length} request{response.skill_gaps.length === 1 ? "" : "s"} sit in the skill-gap list.
            Nobody on this workforce has that skill, so none of the three algorithms is blamed for them.
          </li>
        )}
      </ul>
    </div>
  );
}

export default function App() {
  const [nTechs, setNTechs] = useState(12);
  const [nRequests, setNRequests] = useState(20);
  const [seed, setSeed] = useState(1);
  const [scarcity, setScarcity] = useState(0.4);
  const [weights, setWeights] = useState({ w_distance: 1, w_balance: 2, w_unassigned: 30 });
  const [trap, setTrap] = useState(false);
  const [scenario, setScenario] = useState(null);
  const [response, setResponse] = useState(null);
  const [selectedId, setSelectedId] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const handle = setTimeout(async () => {
      setLoading(true);
      setError("");
      try {
        const scenarioResponse = trap
          ? await fetch(`${API}/api/scenario/trap`)
          : await fetch(`${API}/api/scenario`, {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({
                n_techs: Number(nTechs),
                n_requests: Number(nRequests),
                seed: Number(seed),
                scarcity: Number(scarcity),
              }),
            });
        if (!scenarioResponse.ok) throw new Error("Could not load the scenario.");
        const nextScenario = await scenarioResponse.json();
        const allocated = await fetch(`${API}/api/allocate`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            technicians: nextScenario.technicians,
            requests: nextScenario.requests,
            weights,
          }),
        });
        if (!allocated.ok) throw new Error("Could not allocate this scenario.");
        setScenario(nextScenario);
        setResponse(await allocated.json());
        setSelectedId((current) => current && nextScenario.requests.some((request) => request.id === current) ? current : nextScenario.requests[0]?.id ?? null);
      } catch (err) {
        setError(err.message || "The API is not reachable.");
      } finally {
        setLoading(false);
      }
    }, 300);
    return () => clearTimeout(handle);
  }, [nTechs, nRequests, seed, scarcity, trap, weights]);

  const colors = useMemo(() => {
    const map = new Map();
    scenario?.technicians.forEach((tech, index) => map.set(tech.id, PALETTE[index % PALETTE.length]));
    return map;
  }, [scenario]);

  const best = useMemo(() => bestValues(response?.results), [response]);

  return (
    <div className="app">
      <header>
        <h1>Field service, Bengaluru</h1>
        <p>
          Three algorithms assign the same technicians to the same repair requests.
          The green number is the lowest total cost. Hover any metric to see what it counts.
        </p>
      </header>

      <section className="guide">
        <article>
          <h2>Resources and skills</h2>
          <p>Each technician is a man with one or two of five skills. His first listed skill is the one he was hired for. He can take 2, 3, or 4 jobs, and he works one of three shifts: 08:00–16:00, 09:00–17:00, or 12:00–20:00.</p>
          <ul>
            <li>HVAC — heating, ventilation, and air conditioning</li>
            <li>Electrical, plumbing, carpentry, appliance</li>
          </ul>
        </article>
        <article>
          <h2>Requests</h2>
          <p>A request asks for one of those five skills, has a time window, a duration of 60, 90, or 120 minutes, and a priority from 1 to 5. Skills are drawn only from what this workforce can do. HVAC and plumbing come up more often.</p>
        </article>
        <article>
          <h2>What the knobs change</h2>
          <ul>
            <li>Headcount, demand, scarcity, and seed build a new scenario. The same seed rebuilds the same scenario.</li>
            <li>Scarcity is the chance a technician has one skill instead of two. Higher scarcity means fewer people can cover a job.</li>
            <li>Distance weight makes kilometers matter more. Balance weight charges a technician for each job he already holds. Unassigned weight is the cost of leaving a job open, multiplied by priority.</li>
          </ul>
        </article>
      </section>

      <section className="controls">
        <label>
          Technicians {nTechs}
          <input type="range" min="1" max="40" value={nTechs} onChange={(event) => { setTrap(false); setNTechs(Number(event.target.value)); }} />
        </label>
        <label>
          Requests {nRequests}
          <input type="range" min="0" max="80" value={nRequests} onChange={(event) => { setTrap(false); setNRequests(Number(event.target.value)); }} />
        </label>
        <label>
          Scarcity {Number(scarcity).toFixed(2)}
          <input type="range" min="0" max="1" step="0.05" value={scarcity} onChange={(event) => { setTrap(false); setScarcity(Number(event.target.value)); }} />
        </label>
        <label>
          Seed
          <input type="number" min="0" value={seed} onChange={(event) => { setTrap(false); setSeed(Number(event.target.value) || 0); }} />
        </label>
        <label>
          Distance weight {weights.w_distance}
          <input type="range" min="0" max="5" step="0.5" value={weights.w_distance} onChange={(event) => setWeights({ ...weights, w_distance: Number(event.target.value) })} />
        </label>
        <label>
          Balance weight {weights.w_balance}
          <input type="range" min="0" max="10" step="0.5" value={weights.w_balance} onChange={(event) => setWeights({ ...weights, w_balance: Number(event.target.value) })} />
        </label>
        <label>
          Unassigned weight {weights.w_unassigned}
          <input type="range" min="0" max="80" step="5" value={weights.w_unassigned} onChange={(event) => setWeights({ ...weights, w_unassigned: Number(event.target.value) })} />
        </label>
        <button type="button" onClick={() => setTrap(true)}>Load trap scenario</button>
      </section>

      {error && <p className="error">{error}</p>}
      {loading && <p className="loading">Building the scenario…</p>}

      {scenario && response && (
        <>
          <p className="scenario-note">{scenario.label}. {scenario.technicians.length} technicians, {scenario.requests.length} requests.</p>
          <table className="metrics">
            <thead>
              <tr>
                <th>Metric</th>
                {ALGORITHMS.map((algo) => <th key={algo.id}>{algo.label}</th>)}
              </tr>
            </thead>
            <tbody>
              {METRIC_COLUMNS.map((column) => (
                <tr key={column.id}>
                  <th><span className="tip" tabIndex={0} data-tip={column.tip}>{column.label}</span></th>
                  {ALGORITHMS.map((algo) => {
                    const metrics = response.results[algo.id].metrics;
                    const value = metricValue(metrics, column.id);
                    const winner = value != null && Math.abs(value - best[column.id]) < 1e-6;
                    return (
                      <td key={algo.id} className={winner ? "best" : ""}>
                        <span className="tip" tabIndex={0} data-tip={column.tip}>{metricText(metrics, column.id)}</span>
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
          <section className="metric-note">
            <Behavior response={response} />
          </section>

          <div className="maps">
            {ALGORITHMS.map((algo) => (
              <section className="panel" key={algo.id}>
                <div className="panel-head">
                  <strong>{algo.label}</strong>
                  <span>{money(response.results[algo.id].metrics.total_cost)} cost</span>
                </div>
                <div className="map-frame">
                  <FieldServiceMap
                    scenario={scenario}
                    result={response.results[algo.id]}
                    colors={colors}
                    selectedId={selectedId}
                    onSelect={setSelectedId}
                  />
                </div>
                <p className="reason">
                  {selectedId ? response.results[algo.id].assignments.find((row) => row.request_id === selectedId)?.explanation || response.skill_gaps.find((gap) => gap.request_id === selectedId)?.reason || "Select a request." : "Select a request."}
                </p>
              </section>
            ))}
          </div>
          <p className="legend">
            <span><i className="swatch" style={{ background: "#0f766e" }} /> Technician, color reused on his lines</span>
            <span><i className="swatch" style={{ background: "#86efac" }} /> Served request, larger means higher priority</span>
            <span><i className="swatch" style={{ background: "#b91c1c" }} /> Left unassigned</span>
            <span><i className="swatch" style={{ background: "#f59e0b" }} /> Skill gap</span>
          </p>

          <div className="roster">
            <article className="panel">
              <h2>Technicians</h2>
              <ul>
                {scenario.technicians.map((tech) => (
                  <li key={tech.id}>
                    <span className="swatch" style={{ background: colors.get(tech.id) }} />
                    {tech.name}, {tech.area}: {tech.skills.map(skillLabel).join(" + ")}, {tech.shift_start}–{tech.shift_end}, up to {tech.max_jobs} {tech.max_jobs === 1 ? "job" : "jobs"}
                  </li>
                ))}
              </ul>
            </article>
            <article className="panel">
              <h2>Requests</h2>
              <ul className="job-list">
                {scenario.requests.map((request) => (
                  <li key={request.id}>
                    <button type="button" className={selectedId === request.id ? "active" : ""} onClick={() => setSelectedId(request.id)}>
                      {request.customer} · P{request.priority} · {skillLabel(request.required_skill)} · {request.window_start}–{request.window_end}
                    </button>
                  </li>
                ))}
              </ul>
              {response.skill_gaps.length > 0 && (
                <>
                  <h2>Skill gaps</h2>
                  <ul>
                    {response.skill_gaps.map((gap) => <li key={gap.request_id}>{gap.reason}</li>)}
                  </ul>
                </>
              )}
            </article>
          </div>
        </>
      )}
    </div>
  );
}
