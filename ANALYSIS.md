# When greedy loses, ties, and is still the right tool

Numbers below are from `backend/experiments.py` on this engine (30 seeds unless noted). They are not copied from a plan. The score is total cost. Gap is `(greedy cost - Hungarian cost) / Hungarian cost`.

## The trap

Vikram, in Koramangala, has electrical and plumbing and can take one job. Arjun, in Jayanagar, has plumbing only and can take one job. The leak is next to Vikram and closes at 12:00. The panel is electrical, only Vikram can do it, and it closes at 16:00. Both are priority 5.

| Algorithm | Cost | Urgent misses | Leak | Panel |
|---|---:|---:|---|---|
| Greedy | 150.26 | 1 | Vikram | unassigned |
| Hungarian | 6.27 | 0 | Arjun | Vikram |
| Regret | 6.27 | 0 | Arjun | Vikram |

Greedy takes the leak first because it is the same priority and the window closes sooner. Vikram is the nearest plumber, so the only electrician is gone. The drop costs `30 * 5 = 150`, plus a short trip. Hungarian and regret leave Vikram for the panel and send Arjun to the leak. Regret does this because the panel has only one feasible technician, so its regret is infinite.

## Load, 10 technicians, scarcity 0.4

| Requests | Greedy gap | Regret gap | Runtime ms (greedy / Hungarian / regret) |
|---:|---:|---:|---|
| 8 | 6.39% | 0.02% | 0.2 / 1.2 / 0.8 |
| 16 | 6.40% | 3.32% | 0.4 / 2.6 / 2.6 |
| 24 | 11.01% | 9.89% | 0.5 / 3.6 / 4.8 |
| 32 | 10.70% | 25.43% | 0.6 / 4.4 / 7.6 |

With spare capacity, regret is essentially tied with Hungarian. The greedy gap is largest around 24 requests, when people still have a choice of jobs, and it does not keep climbing once demand exceeds capacity. At 32 requests regret is worse than greedy: it spends its scarce technicians on the requests that look fragile, then has nothing left for the rest.

## Scarcity, 10 technicians, 20 requests

| Scarcity | Greedy gap | Regret gap | Urgent avg km, greedy | Urgent avg km, Hungarian |
|---:|---:|---:|---:|---:|
| 0.10 | 15.85% | 11.85% | 5.79 | 6.43 |
| 0.40 | 9.63% | 6.67% | 5.88 | 6.44 |
| 0.80 | 6.98% | 3.69% | 7.75 | 8.11 |
| 0.95 | 2.58% | 5.99% | 7.67 | 7.94 |

The gap shrinks as skills get scarcer. Hungarian's advantage is choice. When almost everyone has one skill, there is little to reassign. At scarcity 0.95 regret's gap is larger than greedy's, because protecting the only feasible technician is the wrong trade once most jobs already have only one option.

On every scarcity row, greedy's served priority-4 and priority-5 jobs travel fewer kilometers than Hungarian's. That shorter urgent travel is the local choice. It is also why the total cost is higher: the close urgent job can take the person a later job needed.

## Which algorithm to use

Greedy matches Hungarian when each request has an obvious technician and capacity is loose, and it is the one to ship when requests arrive one at a time or the explanation has to be "we took the best person still free." It fails when a flexible technician is also the only person who can do a second job, which is the trap above.

Hungarian is the score to beat whenever the day's requests are known all at once. It is still fast at these sizes (a few milliseconds).

Regret fixes the trap and stays close to Hungarian while there is spare capacity. It is the wrong default once the board is overloaded: at 32 requests its gap was 25%, against greedy's 11%.
