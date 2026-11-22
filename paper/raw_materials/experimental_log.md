## 1. Experimental Setup

*   **Power system:** RTS-GMLC (NREL Reliability Test System, Grid Modernization Lab Consortium update): 73 buses in 3 areas,
    120 lines, 153 modelled generators (CSP, storage, synchronous condensers and the HVDC link excluded; ~250 MW of a
    ~14.5 GW fleet), 2020 hourly day-ahead load, wind, PV, rooftop PV and hydro. Linear generator costs and CO2 rates come
    from the piecewise heat-rate curves (full-load average heat rate x fuel price + VOM). No unit commitment: Pmin is
    relaxed to 0 except nuclear must-run. Line ratings are derated to 0.6x nameplate to create a congested case
    (RTS-GMLC is lightly congested at nameplate). Value of lost load: $10,000/MWh.
*   **Dispatch model:** multi-period DC optimal power flow in PTDF form, solved with HiGHS. It has generator ramps,
    renewable curtailment, involuntary load shedding at VOLL, and the data-center envelope constraints. LMPs and
    locational marginal emissions (LMEs) are computed from the optimal basis (B^T y = c with costs, or with CO2 rates);
    both are checked against finite differences.
*   **Workload:** Azure LLM inference trace 2024 (DynamoLLM, HPCA 2025; CC-BY): one week each of code-assistant
    (16.8M requests) and conversation (27.3M requests) traffic, mapped to local day-of-week x hour (UTC-7). The mean is
    2,511 context / 23 generated tokens per code request and 1,632 / 106 per conversation request. Batch (deferrable)
    work is parameterised: by default it fills 50% of spare server capacity, with deadlines of 4-24 h.
*   **Fleet model:** one replica = 8x H100 SXM (700 W TDP each), 70B-class model, tensor parallel 8, BF16. The simulation
    is iteration-level with chunked-prefill continuous batching (2,048-token budget, max 256 sequences) and roofline
    timing (prefill MFU 0.5, decode MBU 0.6, KV cache 1.33M tokens). Power-cap model: a phase that demands more than the
    cap runs at perf = ((cap - idle)/(demand - idle))^gamma (gamma 1/3 for prefill, 0.15 for decode). Facility PUE 1.2.
    Spare servers stay in hot standby. Parameters not taken from datasheets are marked as assumptions in the config.
*   **SLOs:** p99 TTFT <= 1 s and p99 TPOT <= 100 ms. Replicas are provisioned at 80% of their full-power SLO capacity
    for the hour's peak minute. Power capping may be used for up to 4 full-depth hours per day.
*   **Default facility:** a 300 MW LLM-inference campus (30,864 servers).
*   **Operating modes (all with the same envelope):** inflexible (batch served on arrival, no capping); price-following
    (best response to the day-ahead LMP at its bus from the inflexible run); average-carbon-following (system average
    emission rate); LME-following (locational marginal emissions at its bus); co-optimised (envelope inside the OPF,
    minimising system cost). A 48 h-lookahead rolling horizon commits 24 h per day over 364 days and carries ramp state
    and the actual batch backlog. Congestion cost = cost on the real network minus the cost of the same loads on a copper
    plate.
*   **Sites:** buses 101 (coal-heavy area 1), 122 (wind, area 1), 207 (radial pocket, area 2), 303 (wind pocket,
    area 3) and 313 (load centre, area 3).

## 2. Raw Numeric Data

### 2.1 Grid baseline without a data center (one year)

| line ratings | load (TWh) | gen. cost ($M) | CO2 (Mt) | avg CI (t/MWh) | curtailment (TWh) | congestion rent ($M) | binding line-hours | mean LME (t/MWh) | mean LME spread across buses (t/MWh) |
|---|---|---|---|---|---|---|---|---|---|
| 1.000 | 37.656 | 449.225 | 15.721 | 0.417 | 0.731 | 30.509 | 4114.000 | 0.713 | 0.377 |
| 0.600 | 37.656 | 484.225 | 16.810 | 0.446 | 2.057 | 104.829 | 22328.000 | 0.725 | 1.067 |

Coal steam units have a mean marginal cost of $22.9/MWh at 1.03 tCO2/MWh. Gas combined cycle: $29.0/MWh at 0.40 tCO2/MWh. Coal is cheaper but dirtier, so the marginal emission rate (~0.71 t/MWh) far exceeds the average (~0.42 t/MWh).

### 2.2 Power-capping trade-off per 8-GPU replica (Figure: fig2a_power_cap_tradeoff)

| service | GPU cap (% TDP) | SLO capacity (% of uncapped) | server power at nominal load (kW) | feasible at nominal load |
|---|---|---|---|---|
| code | 50 | 11.50 | 4.58 | False |
| code | 60 | 20.10 | 4.96 | False |
| code | 70 | 88.20 | 5.23 | True |
| code | 80 | 95.10 | 5.48 | True |
| code | 90 | 100.00 | 5.71 | True |
| code | 100 | 100.00 | 5.93 | True |
| conv | 50 | 38.30 | 4.73 | False |
| conv | 60 | 41.80 | 5.34 | False |
| conv | 70 | 86.50 | 5.68 | True |
| conv | 80 | 93.90 | 5.97 | True |
| conv | 90 | 94.40 | 6.25 | True |
| conv | 100 | 100.00 | 6.51 | True |

Batch (throughput-optimal, latency-unconstrained) operation per replica:

| GPU cap (% TDP) | server power (W) | generated tokens/s | J per generated token |
|---|---|---|---|
| 100.00 | 7730.27 | 1283.75 | 6.02 |
| 90.00 | 7244.59 | 1248.22 | 5.80 |
| 80.00 | 6679.08 | 1205.75 | 5.54 |
| 70.00 | 6042.50 | 1155.14 | 5.23 |
| 60.00 | 5391.38 | 1096.59 | 4.92 |
| 50.00 | 4726.23 | 1027.07 | 4.60 |

### 2.3 Flexibility envelope of the 300 MW facility (Figures: fig2b_envelope_week, fig2c_flex_sweeps)

| quantity | value |
|---|---|
| mean draw (MW) | 157.447 |
| peak draw (MW) | 201.871 |
| load factor | 0.525 |
| mean firm power (MW) | 126.345 |
| mean sheddable power within SLO (MW) | 7.703 |
| sheddable share of load | 0.049 |
| mean headroom for deferrable work (MW) | 62.206 |
| deferrable (batch) share of energy | 0.198 |
| batch J/token (at best cap) | 4.602 |

Sensitivity of flexibility to SLO tightness (SLO scale multiplies both TTFT and TPOT targets) and to batch fill:

| sweep | setting | sheddable MW | sheddable share | batch share of energy | mean draw MW |
|---|---|---|---|---|---|
| slo | 0.600 | 0.975 | 0.008 | 0.251 | 129.906 |
| slo | 0.750 | 1.886 | 0.014 | 0.243 | 133.100 |
| slo | 1.000 | 7.703 | 0.049 | 0.198 | 157.447 |
| slo | 1.250 | 10.365 | 0.065 | 0.193 | 160.650 |
| slo | 1.500 | 12.009 | 0.073 | 0.187 | 163.748 |
| slo | 2.000 | 16.953 | 0.102 | 0.184 | 165.689 |
| fill | 0.000 | 7.703 | 0.061 | 0.000 | 126.345 |
| fill | 0.250 | 7.703 | 0.054 | 0.110 | 141.896 |
| fill | 0.500 | 7.703 | 0.049 | 0.198 | 157.447 |
| fill | 0.750 | 7.703 | 0.045 | 0.270 | 172.999 |
| fill | 0.900 | 7.703 | 0.042 | 0.307 | 182.330 |

### 2.4 One-year grid value of the same envelope under different signals (Figures: fig3_value_by_mode, fig4_signal_paradox)

Values are changes versus an inflexible facility at the same bus, as % of the facility's *hosting impact* (what the inflexible facility adds over no data center). Negative = better. Lines at 0.6x, 300 MW facility, 364 days.

Hosting impact of the inflexible facility:

| bus | hosting cost ($M/yr) | hosting CO2 (kt/yr) | facility energy (TWh) | involuntary shedding (GWh) | hosting CO2 intensity (t/MWh) |
|---|---|---|---|---|---|
| 101.000 | 34.377 | 1110.669 | 1.375 | 0.000 | 0.808 |
| 122.000 | 24.650 | 720.090 | 1.375 | 0.000 | 0.524 |
| 207.000 | 63.043 | 535.139 | 1.375 | 155.608 | 0.389 |
| 303.000 | 26.810 | 657.072 | 1.375 | 0.016 | 0.478 |
| 313.000 | 33.667 | 789.986 | 1.375 | 0.000 | 0.574 |

| bus | mode | d_cost_pct_of_hosting | d_co2_pct_of_hosting | d_congestion_cost_pct |
|---|---|---|---|---|
| 101 | price | -2.77 | 0.86 | 0.40 |
| 101 | avg_ci | -1.80 | 0.00 | 0.91 |
| 101 | lme | -1.38 | -5.73 | 0.29 |
| 101 | coopt | -2.78 | 0.88 | 0.39 |
| 122 | price | -6.79 | -3.39 | -2.01 |
| 122 | avg_ci | -3.98 | -2.18 | 0.16 |
| 122 | lme | -5.26 | -11.90 | -1.85 |
| 122 | coopt | -6.66 | -3.08 | -2.04 |
| 303 | price | -8.33 | -3.67 | -3.69 |
| 303 | avg_ci | -3.30 | -0.16 | 0.41 |
| 303 | lme | -6.27 | -12.28 | -3.07 |
| 303 | coopt | -8.39 | -3.22 | -3.81 |
| 313 | price | -3.91 | -2.38 | -0.64 |
| 313 | avg_ci | -3.61 | -0.93 | -0.41 |
| 313 | lme | -2.24 | -9.56 | -0.05 |
| 313 | coopt | -3.91 | -2.38 | -0.68 |

Lever decomposition at bus 313 (shift = batch deferral only; shed = power capping only):

| name | d_cost_pct_hosting | d_emis_pct_hosting | dc_shed_gwh | dc_shift_gwh |
|---|---|---|---|---|
| 313_price | -3.91 | -2.38 | 20.50 | 271.54 |
| 313_avg_ci | -3.61 | -0.93 | 20.50 | 271.26 |
| 313_coopt | -3.91 | -2.38 | 20.50 | 271.24 |
| 313_coopt_shift | -2.11 | -0.84 | 0.00 | 271.22 |
| 313_coopt_shed | -1.81 | -1.53 | 20.50 | 271.87 |
| 313_price_shift | -2.11 | -0.84 | 0.00 | 271.54 |
| 313_avg_ci_shift | -1.84 | 0.47 | 0.00 | 271.26 |

Grid-wide $50/t carbon price at bus 313 (changes vs the inflexible facility on the same carbon-priced grid):

| name | d_cost_musd | d_emissions_kt |
|---|---|---|
| 313_inflexible_cp50 | 0.00 | 0.00 |
| 313_price_cp50 | -1.37 | -22.34 |
| 313_lme_cp50 | -1.34 | -22.45 |
| 313_coopt_cp50 | -1.37 | -22.40 |

Bus 207 (radial, a single 105 MW line at 0.6x) cannot host 300 MW. Involuntary load shedding by mode:

| mode | grid_shed_gwh | d_grid_shed_gwh | late_mwh_h |
|---|---|---|---|
| inflexible | 155.61 | 0.00 | 0.00 |
| price | 143.39 | -12.22 | 50235.07 |
| avg_ci | 222.10 | 66.49 | 0.00 |
| lme | 214.53 | 58.92 | 0.00 |
| coopt | 100.08 | -55.53 | 26830.84 |

### 2.5 Solver study: 24 h joint dispatch at bus 313 (Figure: fig5_solver_benchmark)

Family: 24 h DC-OPF with one facility envelope, 5496 decision variables after completion. 1168 training / 296 held-out-day test / 74 out-of-distribution (600 MW facility, batch fill 0.9) instances, all labelled by HiGHS. Exact methods were timed one instance at a time in one process; learned methods were batched on 4 CPU threads (float64). Every exact-method solution was asserted to have zero line-limit violation and to match the full-LP objective.

| method | split | ms/instance | speedup | mean optimality gap (%) | max violation (MW) | feasible fraction | LP rounds |
|---|---|---|---|---|---|---|---|
| HiGHS (full LP) | test | 243.83 | 1.00 | 0.00 | 0.00 | 1.00 | 1.00 |
| Constraint generation (no learning) | test | 111.02 | 2.20 | 0.00 | 0.00 | 1.00 | 3.01 |
| Frequency screening + CG (no learning) | test | 75.86 | 3.21 | 0.00 | 0.00 | 1.00 | 1.19 |
| Learned screening + CG | test | 93.76 | 2.60 | 0.00 | 0.00 | 1.00 | 2.52 |
| Oracle screening (bound) | test | 35.22 | 6.92 | 0.00 | 0.00 | 1.00 | 1.02 |
| Gauge map (pure) | test | 0.80 | 306.15 | 2566.15 | -0.00 | 1.00 | nan |
| DC3-style | test | 11.35 | 21.48 | 211.07 | 178.36 | 0.00 | nan |
| Predict-correct-project | test | 7.23 | 33.73 | 11268.15 | 0.00 | 1.00 | nan |
| HiGHS (full LP) | ood_600MW | 244.28 | 1.00 | 0.00 | 0.00 | 1.00 | 1.00 |
| Constraint generation (no learning) | ood_600MW | 117.30 | 2.08 | 0.00 | 0.00 | 1.00 | 3.08 |
| Frequency screening + CG (no learning) | ood_600MW | 88.50 | 2.76 | 0.00 | 0.00 | 1.00 | 1.39 |
| Learned screening + CG | ood_600MW | 106.22 | 2.30 | 0.00 | 0.00 | 1.00 | 2.81 |
| Oracle screening (bound) | ood_600MW | 35.17 | 6.95 | 0.00 | 0.00 | 1.00 | 1.01 |
| Gauge map (pure) | ood_600MW | 1.09 | 223.87 | 9066.85 | -0.00 | 1.00 | nan |
| DC3-style | ood_600MW | 8.11 | 30.13 | 170.50 | 523.09 | 0.00 | nan |
| Predict-correct-project | ood_600MW | 7.88 | 31.02 | 20578.43 | 0.00 | 1.00 | nan |

Further solver observations:
*   Completing the system power balance with a single pivot variable concentrates the prediction error of ~5,000
    variables on one bus (about 2,000 MW of error). A distributed-slack completion (every generator moves by the same
    fraction of its range) keeps the slack within 0.15.
*   A closed-form strictly interior anchor exists for 100% of instances (lowest-shedding member of a one-parameter
    family, shedding 2-50% of load). Gauge-map and ray-projection outputs are exactly feasible for arbitrary network
    outputs, but every interior point of this LP must shed load, the only slack, at VOLL. So feasibility restoration is
    very expensive.
*   FSNet-style L-BFGS feasibility seeking (normalised squared violation, MW-scaled variables, restarts) stalls at
    13-40 MW of violation and costs 160-800 ms per instance, slower than HiGHS.
*   The learned screen reaches only 77% recall on unseen days (it overfits). Its flagged rows are a subset of the
    frequency screen's, so the union adds nothing.

### 2.6 Flexible-interconnection map (Figure: fig1_flex_interconnection_map)

Ladder (same fleet, traffic and energy on every rung): L0 inflexible; L1 power cap within SLO; L2/L3/L4 batch deferral 4/12/24 h; L5 capping down to a 2x-relaxed SLO. Hostable = no involuntary load shedding and no missed batch deadline, with the operator co-optimising the envelope. Stage 1: exact screened HiGHS on 48 stress days (4 per month, 48 h windows), which gives exact 'not hostable' answers and exact lower bounds. Stage 2: every flexible entry verified over all 365 days and raised until every day passes.

| minimum level (verified) | 100 MW | 250 MW | 500 MW | 1000 MW |
|---|---|---|---|---|
| L0 inflexible | 68 | 53 | 36 | 23 |
| L1 power-cap | 1 | 1 | 2 | 1 |
| L2 + defer 4 h | 0 | 5 | 8 | 3 |
| L3 + defer 12 h | 0 | 0 | 0 | 2 |
| L5 + SLO x2 | 0 | 1 | 2 | 1 |
| not hostable | 4 | 13 | 25 | 43 |

| size_mw | hostable_inflexible | hostable_with_flexibility | not_hostable |
|---|---|---|---|
| 100 | 68 | 69 | 4 |
| 250 | 53 | 60 | 13 |
| 500 | 36 | 48 | 25 |
| 1000 | 23 | 30 | 43 |

Hosting capacity (sum over the 73 buses of the largest hostable campus, one campus at a time): 35.25 GW inflexible vs 42.90 GW with SLO-feasible flexibility.

Stage 2 raised 10 of 30 flexible entries found by stage 1. Doubling the stress days from 2 to 4 per month changed only 1 of 292 entries, so system-wide stress days miss locally stressful days.

Full-year spot checks (all 365 days) at the phase-3 sites:

| bus | size_mw | map_level | checked_level | failing_days | shed_mwh | late_mwh_h |
|---|---|---|---|---|---|---|
| 101.00 | 100.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 101.00 | 250.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 101.00 | 500.00 | 2.00 | 1.00 | 9.00 | 196.80 | 0.00 |
| 101.00 | 500.00 | 2.00 | 2.00 | 2.00 | 10.76 | 0.00 |
| 122.00 | 100.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 122.00 | 250.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 122.00 | 500.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 122.00 | 1000.00 | 2.00 | 1.00 | 47.00 | 2955.79 | 0.00 |
| 122.00 | 1000.00 | 2.00 | 2.00 | 4.00 | 0.00 | 187.25 |
| 303.00 | 100.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 303.00 | 250.00 | 2.00 | 1.00 | 2.00 | 3.34 | 0.00 |
| 303.00 | 250.00 | 2.00 | 2.00 | 0.00 | 0.00 | 0.00 |
| 313.00 | 100.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 313.00 | 250.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 313.00 | 500.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

## 3. Qualitative Observations

*   **Interactive LLM serving is barely flexible.** Under p99 SLOs, replicas provisioned at 80% of capacity can only be
    capped to ~70% of TDP. Below that, SLO capacity collapses, because the chunked-prefill TPOT floor exceeds 100 ms. Most
    real flexibility comes from deferrable batch work, so "X% flexible" claims implicitly assume a batch share.
*   **Capping is efficient for batch work.** The 50% cap gives the lowest energy per token, so deferring batch work also
    saves energy.
*   **The signal matters more than the flexibility.** With an identical envelope, LME-following cuts hosting emissions
    3-4x more than cost-optimal co-optimisation while keeping most of the cost savings. Average-carbon-following (the
    default in carbon-aware schedulers) is nearly useless, and sometimes counter-productive.
*   **The "lower cost, higher emissions" effect appears exactly where coal is on the margin** (bus 101: cost down,
    emissions up). A grid carbon price aligns all signals.
*   **A single 300 MW facility is close to a price-taker,** so price-following and co-optimisation give almost identical
    outcomes.
*   **Location dominates operation.** The hosting CO2 intensity of the same facility varies from 0.48 to 0.81 t/MWh across
    buses, far more than any operating mode changes it. At the radial bus 207, network-blind carbon signals *increase*
    involuntary shedding while co-optimisation reduces it.
*   **Modest flexibility unlocks substantial hosting capacity.** Most of the gain comes from cheap rungs (power capping,
    4 h deferral). No site needs exactly 24 h deferral as its minimum.
*   **Certifying the map needs full-year verification;** stress days alone were optimistic for marginal sites.
*   **End-to-end hard-constraint learners hit a structural barrier on OPF with load-shedding slack** (the interior-anchor
    problem). Exact constraint screening with constraint generation is the practical way to fast, certifiably optimal
    solutions on this problem. Learning low-cost interior anchors is an open problem.
*   **Limitations:** a single test system (RTS-GMLC) with derated lines; one facility at a time; perfect day-ahead
    foresight; no unit commitment; fleet parameters partly assumed (not yet swept); a simulation study, not a deployment.
