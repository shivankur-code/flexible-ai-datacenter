"""Literature-agent output: references.bib + citation_map.json from *verified* bibliographic metadata only.

Every entry comes from an API record (Crossref / OpenAlex / arXiv) whose title fuzzy-matched the candidate at
>= 90 (see verify2.py). Nothing is typed from memory. Citation keys are fixed per candidate so that the manuscript can
be written before verification finishes; check_cites() then reports any \\cite key with no verified entry.

    python paper/orchestra_run/build_bib.py verified.json
"""
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
KEYS = {
    "DC3: A learning method for optimization with hard constraints": "donti2021dc3",
    "Efficient Memory Management for Large Language Model Serving with PagedAttention": "kwon2023pagedattention",
    "Orca: A Distributed Serving System for Transformer-Based Generative Models": "yu2022orca",
    "Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve": "agrawal2024sarathi",
    "Splitwise: Efficient generative LLM inference using phase splitting": "patel2024splitwise",
    "DynamoLLM: Designing LLM Inference Clusters for Performance and Energy Efficiency": "stojkovic2025dynamollm",
    "POLCA: Power Oversubscription in LLM Cloud Providers": "patel2024polca",
    "Zeus: Understanding and Optimizing GPU Energy Consumption of DNN Training": "you2023zeus",
    "Roofline: an insightful visual performance model for multicore architectures": "williams2009roofline",
    "The IEEE Reliability Test System: A Proposed 2019 Update": "barrows2020rts",
    "Parallelizing the dual revised simplex method": "huangfu2018highs",
    "Flexible Data Centers and the Grid: Lower Costs, Higher Emissions?": "knittel2025flexible",
    "The environmental potential of hyper-scale data centers: Using locational marginal CO2 emissions to guide geographical load shifting": "lindberg2021environmental",
    "Remunerating space–time, load-shifting flexibility from data centers in electricity markets": "zhang2022remunerating",
    "Carbon-Aware Computing for Datacenters": "radovanovic2023carbon",
    "Let's Wait Awhile: How Temporal Workload Shifting Can Reduce Carbon Emissions in the Cloud": "wiesner2021letswait",
    "CarbonScaler: Leveraging Cloud Workload Elasticity for Optimizing Carbon-Efficiency": "hanafy2024carbonscaler",
    "DeepOPF: A Deep Neural Network Approach for Security-Constrained DC Optimal Power Flow": "pan2021deepopf",
    "Computationally Efficient Safe Reinforcement Learning for Power Systems": "tabas2022safe",
    "Low Complexity Homeomorphic Projection to Ensure Neural-Network Solution Feasibility for Optimization over (Non-)Convex Set": "liang2023homeomorphic",
    "FSNet: Feasibility-Seeking Neural Network for Constrained Optimization with Guarantees": "nguyen2025fsnet",
    "Data-Driven Screening of Network Constraints for Unit Commitment": "pineda2020screening",
    "Tackling Climate Change with Machine Learning": "rolnick2022tackling",
    "Recalibrating global data center energy-use estimates": "masanet2020recalibrating",
    "Analysis of marginal carbon intensities in constrained power networks": "ruiz2010marginal",
    "Task-based End-to-end Model Learning in Stochastic Optimization": "donti2017taskbased",
    "Opportunities and challenges for data center demand response": "wierman2014opportunities",
    "2024 United States Data Center Energy Usage Report": "shehabi2024report",
    "Learning to Solve Optimization Problems With Hard Linear Constraints": "li2023loop",
    "Enforcing Policy Feasibility Constraints through Differentiable Projection for Energy Optimization": "chen2021enforcing",
    "Learning Optimal Solutions for Extremely Fast AC Optimal Power Flow": "zamzam2020learning",
    "Characterizing Power Management Opportunities for LLMs in the Cloud": "patel2024characterizing",
    "Sustainable AI: Environmental Implications, Challenges and Opportunities": "wu2022sustainable",
}


def tex(s: str) -> str:
    s = s.replace("\\", " ").replace("&", r"\&").replace("%", r"\%").replace("$", r"\$").replace("#", r"\#").replace("_", r"\_")
    return re.sub(r"\s+", " ", s).strip()


def entry(key: str, d: dict) -> str:
    authors = " and ".join(tex(a) for a in d["authors"]) or "Anonymous"
    venue = d.get("venue") or ""
    typ = (d.get("type") or "").lower()
    title = "{" + tex(d["title"]) + "}"
    f = {"title": title, "author": authors, "year": str(d.get("year") or "")}
    if venue.startswith("arXiv") or d.get("source") == "arxiv" or "arxiv" in venue.lower():
        kind = "misc"
        f["howpublished"] = tex(venue if venue.startswith("arXiv") else "arXiv preprint")
    elif "proceedings" in typ or "proceedings" in venue.lower() or "symposium" in venue.lower() or "conference" in venue.lower():
        kind = "inproceedings"
        f["booktitle"] = tex(venue)
    elif typ in ("journal-article", "article") and venue:
        kind = "article"
        f["journal"] = tex(venue)
    else:
        kind = "misc"
        if venue:
            f["howpublished"] = tex(venue)
    if d.get("doi"):
        f["doi"] = d["doi"]
    body = ",\n".join(f"  {k} = {{{v}}}" for k, v in f.items() if v)  # title already carries one brace pair -> {{...}} keeps case
    return f"@{kind}{{{key},\n{body}\n}}\n"


def check_cites(tex_path: Path, keys: set) -> list:
    src = tex_path.read_text()
    used = set()
    for m in re.finditer(r"\\cite[pt]?\*?\{([^}]*)\}|\\citealp\{([^}]*)\}", src):
        for k in (m.group(1) or m.group(2)).split(","):
            used.add(k.strip())
    return sorted(used - keys)


def main():
    verified = json.load(open(sys.argv[1]))
    bib, cmap = [], {}
    for d in verified:
        key = KEYS.get(d["query"])
        if not key:
            continue
        bib.append(entry(key, d))
        cmap[key] = dict(title=d["title"], authors=d["authors"], year=d["year"], venue=d["venue"], doi=d.get("doi", ""),
                         source=d["source"], match_score=d["score"], abstract=d.get("abstract", ""))
    (HERE / "latex" / "references.bib").write_text("\n".join(bib))
    json.dump(cmap, open(HERE / "citation_map.json", "w"), indent=1)
    missing = check_cites(HERE / "latex" / "paper.tex", set(cmap))
    print(f"{len(cmap)} verified references; cited but unverified: {missing}")


if __name__ == "__main__":
    main()
