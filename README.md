# Resource Allocation Engine

Field service for technicians in Bengaluru. Each technician can take two, three, or four jobs. Greedy, Hungarian, and regret assign the same scenario with the same cost and the same hard rules. The winner is the lowest total cost. Hungarian is the minimum of that cost for this model.

A job is feasible only when the technician has the skill, can arrive before the window closes, and can finish inside his shift. Travel is straight-line distance at 40 km/h, starting from home when the shift starts. Jobs are not chained into a route, so two jobs for one person are priced separately and can overlap.

```
cost = distance_weight * km + balance_weight * jobs_already_held
unserved = unassigned_weight * priority
total = sum of those
```

Defaults are distance 1, balance 2, and unassigned 30. A dropped priority-5 job costs 150.

## Run locally

Requires Python 3.10 or newer and Node.js 18 or newer. The first install and the OpenStreetMap tiles need internet access. No API key is required.

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

The UI sliders build a new scenario from headcount, demand, scarcity, and seed. Moving only the three weights reruns the algorithms on the same scenario. **Load trap scenario** is the hand-built case: greedy gives the nearby plumbing job to the only electrician and leaves the panel open. Hungarian and regret serve both.

## Tests

With the virtual environment active, from this folder:

```bash
pytest
```

`pytest.ini` adds `backend` to the Python path. The sweep behind `ANALYSIS.md` is:

```powershell
# Windows
$env:PYTHONPATH = "backend"
python backend\experiments.py
```

```bash
# macOS or Linux
PYTHONPATH=backend python backend/experiments.py
```

## Approach

Technicians are generated first and never edited to fit demand. Request skills are drawn only from skills that workforce has, skewed toward HVAC and plumbing. A request whose skill nobody has is a skill gap and is excluded from the algorithm scores.

Greedy sorts by priority, then earliest window close, and commits the cheapest feasible technician. Hungarian expands each technician into `max_jobs` slots. Slot k pays the balance cost of k jobs already held, and each request has a dummy column priced at the drop penalty. Regret serves the request with the largest gap between its best and second-best option first. A request with only one feasible technician has infinite regret.

## Assumptions and limitations

- Five skills: HVAC, electrical, plumbing, carpentry, appliance. Technician names are male Indian names.
- Straight-line distance, constant 40 km/h, no road network.
- Jobs are not sequenced. A later job is priced from home, not from the previous job, and two jobs can overlap in time.
- The comparison score is total cost. Urgent kilometers (priority 4 and 5) are reported so a short trip is not mistaken for a better plan.
