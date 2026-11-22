# Research Plan — How Flexible Must an AI Data Center Be?

**SLO-grounded flexibility envelopes, network-constrained grid value, and feasibility-guaranteed fast dispatch for LLM inference fleets.**

Target reader: Prof. Priya Donti (MIT EECS / LIDS). The project is meant to fill the
`X% / Y% / Z% / N×` placeholders and "Figure 1" in the research brief with real,
reproducible numbers.

---

## 1. Why this, and why it would be worth her time

The brief asks: *when AI data centers on a transmission network flex within their latency
SLOs, how much do they reduce cost, congestion and emissions under OPF, and can a
constraint-enforcing learned solver plan joint fleet–grid dispatch fast enough to be useful?*

Answering that as stated would be useful but incremental. Three things already exist:

| Already done | By whom (verify before writing) | What it leaves open |
|---|---|---|
| Geographic/temporal DC load shifting guided by **locational marginal emissions** in OPF | Lindberg, Roald et al. (≈2021–22); Zhang & Zavala (space-time flexibility) | Flexibility is an **assumed %** of load, not derived from what a serving fleet can do without breaking SLOs |
| Price-following flexibility can **cut cost but raise emissions** | Knittel, Senga, Wang, NBER WP 34065 (2025) | Mostly a market/average view; the **network** (congestion, locational marginal units) is not the lens |
| Constraint-enforcing learned optimizers (DC3, FSNet, gauge maps, DeepOPF variants) | Donti, Rolnick, Kolter (ICLR '21); Nguyen & Donti (FSNet); Tabas & Zhang; Pan et al. | A real application where the **speed actually matters**: a plain DC-OPF LP solves in milliseconds with HiGHS, so "N× faster" on its own does not convince her group |

**The advancement (one sentence):**
> We turn LLM-serving SLOs into **polyhedral flexibility envelopes** that plug directly into
> multi-period DC-OPF. We show the grid value of that flexibility is **locational**:
> price-following reproduces the Knittel cost-down/emissions-up effect, and co-optimization
> removes it. We then use a **feasibility-guaranteed learned solver** to make a question
> tractable that HiGHS alone cannot sweep: *the minimum SLO-feasible flexibility an AI data
> center must offer to be hosted at each bus without network upgrades*, a
> **flexible-interconnection map**.

Why this hooks her specifically:
- **Hard constraints are central, not decorative.** The learned solver must respect grid
  physics *and* the fleet's SLO envelope with zero violations. That is DC3/FSNet's whole thesis.
- **The speed-up has a job to do.** Hosting-capacity maps need on the order of buses × DC
  sizes × flexibility levels × 8,760 h × weather scenarios ≈ 10⁶–10⁸ OPF solves. That is
  where amortized learning beats a solver called in a loop, and the claim is honest.
- **The policy hook is timely.** Large-load interconnection queues and "flexible / curtailable
  interconnection" rules for data centers are a live 2025–26 debate (e.g. Duke's
  *Rethinking Load Growth* headroom estimate, Texas large-load curtailment rules, the
  Emerald AI field demo). A per-bus "flexibility you must offer" map speaks to it directly.
- **It plays to Shivankur's real edge.** He brings an operator's view of fleets: drains,
  capacity limits, and serving SLOs (vLLM / llm-d / kvpilot work). Most grid papers assume
  flexibility; this one derives it.

---

## 2. Research questions → the brief's placeholders

| RQ | Question | Fills |
|---|---|---|
| **RQ1 Envelope** | Given a workload mix (interactive chat, API, batch/offline) and SLOs (TTFT, TPOT, completion deadline), what hourly power envelope (floor, ceiling, ramp, deferrable energy with deadline) can a GPU inference fleet offer **without SLO violation**? How does it depend on the batch-job share and SLO tightness? | "SLO-aware" claim; Fig. 2 |
| **RQ2 Grid value** | On a network-constrained test grid with renewables, how much do **inflexible / price-following / carbon-signal-following / co-optimized** data centers change total cost, congestion rent, renewable curtailment and **true** emissions (re-dispatch accounting)? | **X%** (congestion cost), **Y%** (emissions); Fig. 3 |
| **RQ3 Paradox** | When and where does price-following raise emissions (Knittel), and does the network (congestion, the marginal unit at that bus) explain it? Does co-optimization with locational marginal emissions remove it? | Fig. 4 |
| **RQ4 Learned solver** | Can a DC3-style (completion + correction), FSNet-style (feasibility-seeking) or gauge-map solver match the joint multi-period LP within Z% with **zero** grid/SLO violations, at N× the speed of warm-started HiGHS? | **Z%**, **N×**, **zero** |
| **RQ5 Headline** | For each candidate bus and DC size, what is the **minimum SLO-feasible flexibility** that avoids congestion-driven curtailment or new constraint violations across a year of weather? | **Figure 1** (the hosting-capacity vs required-flexibility map) |

---

## 3. Method

### 3.1 Fleet → flexibility envelope (`gridflex.fleet`, `gridflex.envelope`)
- **Workload.** Azure public LLM inference traces (2023 Splitwise, 2024 releases in
  `AzurePublicDataset`) for interactive arrivals. Batch and deferrable work is synthetic but
  parameterized (share 0–50%, deadlines 1–24 h).
- **Serving and power model.** Per-GPU throughput, latency and power as functions of batch
  size and frequency or power cap, using published measurements (ML.ENERGY / Zeus,
  DynamoLLM, POLCA, Splitwise). Queueing is simulated per hour to check TTFT/TPOT p99 against
  the SLO.
- **Flexibility levers.** (a) power capping / frequency scaling (shed, at some latency cost);
  (b) temporal deferral of batch work (shift); (c) optional spatial routing across 2–3 DC
  sites at different buses (shift in space, subject to a network-RTT SLO).
- **Output.** For each hour *t* and site *d*, a polytope over power p_{d,t}:
  `P_min ≤ p ≤ P_max`, ramp limits, and deferrable-energy constraints
  `Σ_{τ≤t} served_τ ≥ Σ_{τ≤t−deadline} arrived_τ`. It is linear, so it plugs into the LP
  exactly. **This is the key modeling step:** SLOs become linear constraints.

### 3.2 Grid (`gridflex.grid`, `gridflex.opf`)
- **Primary:** NREL **RTS-GMLC** (73-bus; real fuel types, heat rates and hourly wind/solar/
  load for a year), so emissions are physically meaningful.
- **Scale check:** **IEEE-118** with NREL's high-renewable extension (hourly profiles and
  generator types), as in the brief.
- **Model:** multi-period DC-OPF in PTDF form, 24 h horizon rolled over the year. It includes
  line limits, ramps, renewable curtailment, the DC envelope constraints, and an optional
  carbon price. Formulated in `linopy`/PyPSA or plain `scipy.sparse` + `highspy`.
- **Emissions:** true emissions = Σ gen × rate from the *re-dispatched* solution, plus
  locational marginal emissions (LMEs) from the duals/sensitivities for signal construction.

### 3.3 Data-center operating modes compared
1. **Inflexible:** p = expected load.
2. **Price-following:** the DC schedules against day-ahead LMPs from run 1, then the grid re-dispatches.
3. **Carbon-signal-following:** the same with *average* CI, and separately with LME.
4. **Co-optimized:** the joint LP (system cost, optionally plus a carbon price).

Modes 2–3 are bilevel in spirit but solved sequentially (DC best-responds to the signal, grid
re-dispatches). That sequential solve is exactly how the Knittel effect shows up.

### 3.4 Feasibility-guaranteed learned solver (`gridflex.solvers`)
Input: (load, wind, solar, DC arrivals, envelope parameters). Output: full 24 h dispatch.
- **DC3:** predict a subset of variables; *complete* the rest through the power-balance
  equalities (PTDF makes this cheap: one slack per period); *correct* inequality violations
  by unrolled gradient steps.
- **FSNet-style:** a network prediction followed by a feasibility-seeking inner solve.
- **Gauge map / homeomorphic projection:** the feasible set is a polytope for each instance,
  so map the network output into it in closed form, using an interior point (the inflexible
  dispatch plus slack). This guarantees feasibility in one forward pass.
- **Baselines:** cold HiGHS, warm-started HiGHS, and an "LP-repair" step (learned guess
  followed by one HiGHS solve).
- **Report:** optimality gap distribution, maximum and mean violation (must be 0 up to a
  tolerance), wall-clock time on CPU and GPU with batching, and out-of-distribution behavior
  (an unseen season, a larger DC).

### 3.5 Headline experiment: the flexible-interconnection map (RQ5)
For each candidate bus b and DC size S ∈ {100, 300, 1000 MW}, sweep the batch share and SLO
slack, which sets the envelope width. Find the **minimum flexibility** such that, across 8,760 h
× K weather scenarios:
- there is no load shedding;
- no new line overloads appear;
- curtailment and cost increase stay under a threshold.

The learned solver runs the full sweep. HiGHS **verifies** a random subset and every point on
the boundary of the map, so the published map is exact where it matters.

---

## 4. Metrics
Total system cost; congestion rent; renewable curtailment (MWh); true CO₂ (t); LMP/LME
spread; SLO violation rate and p99 TTFT/TPOT; solver optimality gap, violations and runtime.

## 5. Planned figures
1. **Figure 1 (headline):** a map of the grid topology, with buses colored by the minimum
   SLO-feasible flexibility (% of peak, or "max batch share needed") required to host an
   S-MW AI data center.
2. **Envelope:** flexibility (MW and MWh shiftable) vs batch share × SLO tightness.
3. **Value by mode:** cost / congestion / curtailment / emissions for the 4 modes.
4. **Paradox:** emissions change of price-following vs co-optimized, by bus and season,
   annotated with the marginal fuel.
5. **Solver frontier:** speed vs optimality gap vs violations (DC3 / FSNet / gauge / HiGHS).

## 6. Phases

| Phase | Output | Est. |
|---|---|---|
| P0 Scaffold | repo, plan, package layout | done |
| P1 Grid | RTS-GMLC loader, multi-period DC-OPF in HiGHS, LMP/LME extraction, unit tests (balance, line limits) | **done**. IEEE-118 moved to P5: pandapower's case118 has placeholder line ratings, so it needs NREL's extended-118 ratings and profiles |
| P2 Fleet | trace loader, serving/power model, per-hour SLO check, envelope polytope export | **done**. Fig. 2a–c. Interactive shed is about 5% of load; batch shift scales with batch share |
| P3 Value | 4 operating modes, a year rolled, Figs 3–4, **X% and Y% filled** | **done**. 5 sites × 6 modes, 48 h lookahead, copper-plate congestion cost. The signal matters more than flexibility; bus 207 is not hostable |
| P4 Solver | dataset generation, DC3 + gauge (+ FSNet), Fig. 5, **Z% and N× filled** | **done**, mostly negative. Exact screening + constraint generation gives 0% gap at 3.2x (6.9x bound); end-to-end learners are infeasible or 2,566–11,268% above optimal. P5 runs on exact screened HiGHS |
| P5 Map | RQ5 sweep, **Figure 1** | **done**. 73 buses × 4 sizes, stress-day screening + full-year verification. Flexibility raises 500 MW hostable sites 36 → 48 and total hosting capacity +22%. IEEE-118 not run (needs NREL-118 ratings) |
| P6 Handoff | `paper/claims.md`, figures, CSVs, `related_work.md`, `refs.bib` → paper-orchestra | 0.5 d |

**Minimum viable result** (if time is short): P1 + P3 on RTS-GMLC with a simple envelope
gives X%, Y% and the paradox figure. That alone is enough to update the brief.

## 7. Honest scoping
- This is a simulation study on public test systems, not a deployment. The paper says so.
- Envelope power and latency parameters come from published measurements, not internal Azure
  data. Nothing proprietary from Microsoft goes in the repo.
- If warm-started HiGHS turns out fast enough for the whole RQ5 sweep, report that honestly.
  The learned-solver contribution then becomes "guaranteed-feasible amortization for
  scenario-scale studies plus OOD analysis", not a raw speed-up claim.
- Before writing, check the novelty claim (SLO-derived envelopes × network-constrained value
  × learned flexible-interconnection map) against recent work, especially Roald's group,
  Zavala's group, Google's carbon-intelligent computing, and Emerald AI.

## 8. Handoff to paper-orchestra
`paper/` will contain: `abstract_seed.md`, `claims.md` (every number links to a CSV row and a
figure), `figures/`, `related_work.md`, `refs.bib`, and `results/` CSVs. paper-orchestra
turns these into the readable paper, and the same numbers go into the two-page brief.
