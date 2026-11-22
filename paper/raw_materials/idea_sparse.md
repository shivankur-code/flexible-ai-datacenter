## Problem Statement
AI data centers are becoming some of the largest new loads on power grids, and interconnection queues are filling with
requests for 100 MW to 1 GW inference campuses. "Flexible" or curtailable interconnection is often proposed as the way
to connect them faster: a campus that can reduce or shift its consumption could be hosted at locations the grid could
not otherwise serve. But three things are missing from how that flexibility is usually studied:

1. **Flexibility is assumed, not derived.** Studies typically posit that some percentage of a data center's load can be
   shifted or shed. For an LLM-inference fleet, the real flexibility is bounded by latency service-level objectives
   (time-to-first-token and time-per-output-token percentiles) and by how much deferrable batch work exists. Nobody has
   connected those serving constraints to grid models.
2. **The grid is treated as a price or carbon signal, not a network.** Flexibility studies that ignore transmission
   constraints cannot say where a facility can connect, nor whether following a price or average-carbon signal helps or
   hurts once congestion and locational marginal units are taken into account. Recent economic work has shown that
   signal-following flexible data centers can lower costs while raising emissions.
3. **Fast solvers with hard guarantees are needed but unproven at this scale.** Learning-to-optimize methods that enforce
   hard constraints (completion and correction, gauge maps, feasibility seeking) promise fast feasible solutions. Joint
   fleet-grid dispatch is a multi-period LP with thousands of variables, deadline constraints and load-shedding slack,
   and it is unclear whether such methods are useful there.

## Core Hypothesis
- An inference fleet's grid flexibility can be derived from first principles (serving physics plus latency SLOs) as a
  *linear* hourly envelope: firm power, power sheddable by GPU power capping within the SLO, and deferrable batch energy
  with deadlines. This envelope plugs exactly into multi-period DC optimal power flow.
- Under transmission constraints, the *signal* that steers this flexibility matters more than the amount of flexibility.
  A network-aware marginal-emissions signal should outperform price-following and average-carbon-following, which can
  even be counter-productive where coal sits on the margin.
- Modest, SLO-feasible flexibility (power capping and a few hours of deferral) should materially expand the set of buses
  where a large campus can be hosted without load shedding. That map can be computed and certified exactly.
- End-to-end hard-constraint learners may struggle on this problem for a structural reason: every strictly interior
  point of an OPF with value-of-lost-load slack sheds load, so feasibility restoration toward an interior point is costly.

## Proposed Methodology (High-Level Technical Approach)
**1. SLO-grounded flexibility envelopes.** Simulate an LLM-inference replica (8-GPU server, 70B-class model) at
iteration level with chunked-prefill continuous batching, a roofline time model and a physically consistent GPU
power-cap model. Drive it with the public Azure LLM inference trace (2024; code-assistant and conversation services).
By bisection, find each replica's p99-SLO capacity under each power cap. Aggregate to a facility (100 MW to 1 GW) to
obtain hourly linear envelopes: firm power, sheddable power (lowest cap that still meets the SLO at the hour's
peak-minute load, with a daily budget), and deferrable batch energy with deadline constraints.

**2. Joint fleet-grid dispatch.** Run multi-period DC-OPF in PTDF form on the RTS-GMLC test system (73 buses, real fuel
types and heat rates, hourly 2020 renewables), with lines derated to create congestion. Use a 48-hour-lookahead rolling
horizon over a full year and solve with HiGHS. Compute locational marginal prices and locational marginal emissions from
the optimal basis. Compare operating modes that use the *same* envelope but different signals: inflexible,
price-following, average-carbon-following, LME-following and grid co-optimised. True emissions always come from the
grid's re-dispatch.

**3. Flexible-interconnection map.** For every bus and campus size, find the lowest rung of a flexibility ladder at which
the campus can be served with no involuntary load shedding and no missed batch deadline. The ladder runs: inflexible,
power cap, defer 4/12/24 h, relaxed-SLO capping; every rung has the same energy. Use a two-stage exact procedure: screen
on stress days (which gives exact lower bounds and exact infeasibility), then verify every flexible entry over all 365
days.

**4. Solver study.** Pose the 24-hour joint problem as a differentiable LP family. Compare DC3-style completion and
correction, a gauge map around a closed-form strictly interior anchor, predict-correct-project (closed-form ray
projection), FSNet-style feasibility seeking, and exact constraint screening with constraint generation (frequency-based
and learned).

## Expected Contribution
- The first SLO-grounded, linear flexibility envelope for LLM-inference fleets, showing that interactive serving offers
  little flexibility and that real flexibility depends on deferrable batch work.
- Evidence that, under network constraints, the steering signal matters more than the flexibility itself. This
  reproduces and explains the "lower cost, higher emissions" effect in network terms.
- A certified flexible-interconnection map showing how much hosting capacity SLO-feasible flexibility unlocks.
- A diagnosis of why hard-constraint learning struggles on OPF with load-shedding slack (the interior-anchor barrier),
  and the finding that exact constraint screening is the practical route to fast, certifiably optimal solutions.
