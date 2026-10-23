"""Seeded Bengaluru scenarios. The workforce is created first and not edited to fit demand."""

import random

from app.models import Request, Scenario, Technician

SKILLS = ("hvac", "electrical", "plumbing", "carpentry", "appliance")
SKILL_WEIGHTS = {
    "hvac": 3.0,
    "plumbing": 3.0,
    "electrical": 1.0,
    "carpentry": 1.0,
    "appliance": 1.0,
}
SHIFTS = (("08:00", "16:00"), ("09:00", "17:00"), ("12:00", "20:00"))
LOCALITIES = (
    ("Koramangala", 12.9352, 77.6245),
    ("Indiranagar", 12.9784, 77.6408),
    ("Whitefield", 12.9698, 77.7500),
    ("Jayanagar", 12.9308, 77.5838),
    ("Malleshwaram", 13.0035, 77.5647),
    ("HSR Layout", 12.9121, 77.6446),
    ("Hebbal", 13.0358, 77.5970),
    ("JP Nagar", 12.9077, 77.5851),
    ("Marathahalli", 12.9591, 77.6974),
    ("Rajajinagar", 12.9982, 77.5550),
    ("Electronic City", 12.8456, 77.6603),
    ("Yelahanka", 13.1007, 77.5963),
)
NAMES = (
    "Aarav", "Aditya", "Anil", "Arjun", "Bharat", "Deepak", "Dev", "Farhan",
    "Gaurav", "Hari", "Harsh", "Imran", "Karan", "Karthik", "Manoj", "Mohit",
    "Naveen", "Nikhil", "Omkar", "Pankaj", "Pranav", "Rahul", "Rajesh", "Ravi",
    "Rohan", "Sanjay", "Suresh", "Tarun", "Varun", "Vikram", "Vivek", "Yash",
)


def _jitter(rng: random.Random, lat: float, lng: float) -> tuple[float, float]:
    return lat + rng.uniform(-0.012, 0.012), lng + rng.uniform(-0.012, 0.012)


def generate(n_techs: int, n_requests: int, seed: int, scarcity: float) -> Scenario:
    rng = random.Random(seed)
    technicians: list[Technician] = []
    for index in range(n_techs):
        area, lat, lng = LOCALITIES[index % len(LOCALITIES)]
        lat, lng = _jitter(rng, lat, lng)
        skill_count = 1 if rng.random() < scarcity else 2
        skills = rng.sample(list(SKILLS), skill_count)
        shift = SHIFTS[rng.randrange(len(SHIFTS))]
        technicians.append(
            Technician(
                id=f"t{index + 1}",
                name=NAMES[index % len(NAMES)],
                lat=round(lat, 6),
                lng=round(lng, 6),
                area=area,
                skills=skills,
                shift_start=shift[0],
                shift_end=shift[1],
                max_jobs=rng.choice((2, 3, 4)),
            )
        )

    offered = sorted({skill for tech in technicians for skill in tech.skills})
    requests: list[Request] = []
    if offered:
        weights = [SKILL_WEIGHTS[skill] for skill in offered]
        for index in range(n_requests):
            area, lat, lng = LOCALITIES[rng.randrange(len(LOCALITIES))]
            lat, lng = _jitter(rng, lat, lng)
            window_start_hour = rng.randint(8, 14)
            window_hours = rng.choice((3, 4))
            requests.append(
                Request(
                    id=f"r{index + 1}",
                    customer=f"Household {index + 1}",
                    lat=round(lat, 6),
                    lng=round(lng, 6),
                    area=area,
                    required_skill=rng.choices(offered, weights=weights, k=1)[0],
                    window_start=f"{window_start_hour:02d}:00",
                    window_end=f"{window_start_hour + window_hours:02d}:00",
                    duration=rng.choice((60, 90, 120)),
                    priority=rng.randint(1, 5),
                )
            )
    return Scenario(
        label=f"{n_techs} technicians, {n_requests} requests, scarcity {scarcity:.2f}",
        seed=seed,
        technicians=technicians,
        requests=requests,
    )


def trap_scenario() -> Scenario:
    """Vikram is nearest for plumbing and the only electrician. Each can take one job.

    Greedy serves the earlier plumbing job and strands the panel. Hungarian and
    regret serve both.
    """
    technicians = [
        Technician(
            id="vikram",
            name="Vikram",
            lat=12.9352,
            lng=77.6245,
            area="Koramangala",
            skills=["electrical", "plumbing"],
            shift_start="08:00",
            shift_end="18:00",
            max_jobs=1,
        ),
        Technician(
            id="arjun",
            name="Arjun",
            lat=12.9308,
            lng=77.5838,
            area="Jayanagar",
            skills=["plumbing"],
            shift_start="08:00",
            shift_end="18:00",
            max_jobs=1,
        ),
    ]
    requests = [
        Request(
            id="leak",
            customer="Household leak",
            lat=12.9370,
            lng=77.6260,
            area="Koramangala",
            required_skill="plumbing",
            window_start="09:00",
            window_end="12:00",
            duration=60,
            priority=5,
        ),
        Request(
            id="panel",
            customer="Household panel",
            lat=12.9500,
            lng=77.6245,
            area="Koramangala",
            required_skill="electrical",
            window_start="09:00",
            window_end="16:00",
            duration=60,
            priority=5,
        ),
    ]
    return Scenario(
        label="Trap: greedy strands the only electrician",
        seed=None,
        technicians=technicians,
        requests=requests,
    )
