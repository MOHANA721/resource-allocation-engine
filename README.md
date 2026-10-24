# Resource Allocation Engine

Field service for technicians in Bengaluru. Each technician can take two, three, or four jobs. Greedy, Hungarian, and regret assign the same scenario with the same cost and the same hard rules. The winner is the lowest total cost. Hungarian is the minimum of that cost for this model.

A job is feasible only when the technician has the skill and a free job slot, can arrive before the window closes, and can finish inside his shift. Travel is at 40 km/h, starting from home when the shift starts.

**Distances** are road distances from the public [OSRM](https://project-osrm.org/) server (OpenStreetMap, no API key) by default. If OSRM cannot be reached, the engine uses straight line × 1.35 and the note under the scenario says so. The UI's **Distance source** selector also offers straight line × 1.35 directly. Road tables are cached per scenario, so moving a slider does not refetch. To use your own OSRM server, set `RAE_OSRM_URL`.

```
cost = distance_weight * km + balance_weight * jobs_already_held
unserved = unassigned_weight * priority
total = sum of those
```

Defaults are distance 1, balance 2, and unassigned 30. A dropped priority-5 job costs 150. All three algorithms leave a job open when serving it costs more than its penalty. With **Always serve P5** on (the default), a priority-5 job is never left open while someone can take it.

Each technician stops at `max_jobs`. A job is priced and time-checked from home, so two jobs for one person can overlap.

## Report

[`REPORT.html`](REPORT.html) is the technical report and algorithm comparison. Open it in any browser; it works offline. It covers the design, hard and soft constraints, the three algorithms, worked case studies (including when each algorithm fails), every adjustable control and its effect, every metric, experimental results, test coverage, and limitations with proposed solutions.

## Run locally

Requires Python 3.10 or newer and Node.js 18 or newer. The first install, the OpenStreetMap tiles, and road distances need internet access; without it, choose Straight line × 1.35, or let the engine fall back to it. No API key is required.

From this folder, on Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:PYTHONPATH = "backend"
uvicorn app.main:app --port 8000
```

On macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=backend
uvicorn app.main:app --port 8000
```

In a second terminal, on any system:

```bash
cd frontend
npm install
npm run dev
```

- API: http://127.0.0.1:8000/docs
- UI: http://localhost:5173

The UI sliders build a new scenario from headcount, demand, scarcity, and seed. Moving the three weights, the distance selector, or the P5 checkbox reruns the algorithms on the same scenario. **Load trap scenario** is the hand-built case: greedy gives the nearby plumbing job to the only electrician and leaves the panel open. Hungarian and regret serve both.

## Tests

With the virtual environment active, from this folder:

```bash
pytest
```

`pytest.ini` adds `backend` to the Python path. The backend suite checks:

- Hungarian against a brute-force search on small scenarios, with and without priority-5 protection.
- Every hard rule (skill, job limit, arrival time, shift end), and how each weight and setting changes each algorithm.
- Distance sources: OSRM, the fallback on a timeout, block requests for large scenarios, and caching. OSRM is replaced by a fake, so the tests run offline.
- That incremental regret gives the same plans as recomputing every round.
- Metric values worked by hand on the trap, explanation text, and repeatability.
- API input validation and the largest slider setting.

The UI unit tests cover the metric table helpers (formatting, hover tips, and which value is marked best). The browser tests use Playwright. They start the API and the UI themselves, or reuse them if already running, then check the distance note, the trap, and the P5 checkbox in Chromium.

```bash
cd frontend
npm test
npx playwright install chromium   # first time only
npm run test:e2e
```

The experiment sweep behind the results in `REPORT.html` is:

```powershell
# Windows
$env:PYTHONPATH = "backend"
python backend\experiments.py
```

```bash
# macOS or Linux
PYTHONPATH=backend python backend/experiments.py
```

The sweeps use straight line × 1.35 so they are reproducible offline. The final comparison of distance sources calls OSRM.

## Approach

Technicians are generated first and never edited to fit demand. Request skills are drawn only from skills that workforce has, skewed toward HVAC and plumbing. A request for a skill nobody has is rejected.

Greedy sorts by priority, then earliest window close, and commits the cheapest feasible technician. Hungarian expands each technician into `max_jobs` slots. Slot k pays the balance cost of k jobs already held, and each request has a dummy column priced at the drop penalty, plus a large guard for protected priority-5 jobs. Regret takes the highest priority first; within a priority it serves the request with the largest gap between its best and second-best option. A request with only one feasible technician has infinite regret. Its option lists are built once and updated only for the technician just assigned.

## Assumptions and limitations

- Five skills: HVAC, electrical, plumbing, carpentry, appliance. Technician names are male Indian names.
- Constant 40 km/h on road distances; no live traffic. The public OSRM server is rate-limited, so a first fetch for 40 technicians and 80 requests takes about 7 seconds.
- Jobs are not sequenced. A later job is priced from home, not from the previous job, and two jobs can overlap in time. A job limit counts jobs, not hours.
- The comparison score is total cost. Urgent kilometers (priority 4 and 5) are reported so a short trip is not mistaken for a better plan.
