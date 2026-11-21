# flexible-ai-datacenter

**How flexible must an AI data center be?** We study SLO-grounded flexibility of LLM inference
fleets, its value under network-constrained optimal power flow, and feasibility-guaranteed
learned dispatch.

- **Fleet → envelope.** Turn LLM-serving SLOs (TTFT, TPOT, batch deadlines) into linear
  hourly flexibility envelopes: power floor and ceiling, ramp limits, deferrable energy.
- **Grid value.** Run multi-period DC-OPF on RTS-GMLC and IEEE-118 with renewables. Compare
  inflexible, price-following, carbon-following and co-optimized data centers on cost,
  congestion, curtailment and *true* emissions.
- **Learned solver.** Use DC3-style, FSNet-style and gauge-map solvers with zero
  grid/SLO violations, benchmarked against HiGHS.
- **Headline.** Build a *flexible-interconnection map*: the minimum SLO-feasible flexibility
  an AI data center must offer to be hosted at each bus.

See **[PLAN.md](PLAN.md)**.

## Status
**Phase 1 done (grid).** The RTS-GMLC loader, multi-period DC-OPF (PTDF + HiGHS)
with data-center envelopes, and LMPs/LMEs from the optimal basis are in place and
tested. Fleet, envelope and learned-solver modules are still stubs.

```bash
pip install -e '.[dev]'
python -m gridflex.datasets               # fetch RTS-GMLC (pinned commit) into data/raw/
pytest -q                                 # 11 tests: toy 3-bus by hand + RTS feasibility, FD-checked LMP/LME
python experiments/p1_baseline_year.py    # full-year baseline -> results/p1/
```

### Phase 1 baseline (RTS-GMLC 2020, no data center, 366 day-ahead OPFs)

| | nameplate ratings | ratings x0.6 (congested case) |
|---|---|---|
| Load | 37.7 TWh | 37.7 TWh |
| Generation cost | $449 M | $484 M |
| CO2 | 15.7 Mt (0.417 t/MWh) | 16.8 Mt (0.446 t/MWh) |
| Renewable curtailment | 0.73 TWh | 2.06 TWh |
| Congestion rent | $31 M | $105 M |
| Binding line-hours | 4,114 | 22,328 |
| Mean LME / mean LME spread across buses | 0.71 / 0.38 t/MWh | 0.72 / 1.07 t/MWh |
| Wall time (laptop CPU) | 77 s | 97 s |

Takeaways for later phases:
- The *marginal* emission rate (about 0.71 t/MWh) is far above the *average*
  (about 0.42). Coal is cheaper than gas combined cycle here, so price-following
  load tends to land on coal. This is the mechanism behind the Knittel effect.
- At nameplate ratings RTS-GMLC is only lightly congested. Congestion studies
  use lines derated to 0.6x and always report the scale.
- Each 24 h OPF takes about 0.25 s in HiGHS, so a learned solver has to justify
  itself on scenario-scale sweeps (RQ5), not on single solves.

## Layout
```
src/gridflex/
  fleet.py       # SLO-aware LLM inference fleet simulator
  envelope.py    # fleet -> hourly flexibility polytope
  grid.py        # RTS-GMLC / IEEE-118 loaders, renewables, emission rates
  opf.py         # multi-period DC-OPF (PTDF) with DC envelopes, HiGHS
  modes.py       # inflexible / price- / carbon-following / co-optimized
  emissions.py   # true emissions + locational marginal emissions
  solvers/       # DC3, FSNet-style, gauge-map learned solvers
experiments/     # one script per RQ -> results/*.csv + paper/figures
data/raw/        # download scripts only (data not committed)
paper/           # claims, figures, related work -> paper-orchestra
```
