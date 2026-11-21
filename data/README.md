# Data

`data/raw/` is gitignored. `python -m gridflex.datasets` fills it.

| Dataset | Where | Used for |
|---|---|---|
| NREL **RTS-GMLC** (pinned commit), grid + 2020 hourly load/wind/solar/hydro | `data/raw/rts_gmlc/` | `gridflex.grid` |
| **Azure LLM inference trace 2024** (code + conversation, 1 week each, 44M requests) | raw: `data/raw/azure_llm/` (1.8 GB). Committed reductions: `data/processed/` | `gridflex.workload`, `gridflex.fleet` |

`data/processed/` is derived from the Azure Public Dataset (CC-BY). Please cite
Stojkovic, Zhang, Goiri, Torrellas, Choukse, *DynamoLLM: Designing LLM Inference
Clusters for Performance and Energy Efficiency*, HPCA 2025. Rebuild it with
`python -m gridflex.datasets --azure`.

- `azure_llm_2024_per_minute.csv`: requests, context tokens and generated tokens per minute, per service (UTC)
- `azure_llm_2024_token_sample.csv`: 92k (context, generated) token pairs sampled uniformly from the trace

Planned: the IEEE-118 extended high-renewable case (NREL) for the phase 5 scale check.
