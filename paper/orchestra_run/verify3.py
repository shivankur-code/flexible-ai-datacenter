import json, re, html, sys, requests
from thefuzz import fuzz
H = {"User-Agent": "gridflex-paper-verifier/0.3"}
tries = [  # (candidate title as in candidates.txt, kind, identifier)
 ("DC3: A learning method for optimization with hard constraints", "arxiv", "2104.12225"),
 ("Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve", "arxiv", "2403.02310"),
 ("Roofline: an insightful visual performance model for multicore architectures", "doi", "10.1145/1498765.1498785"),
 ("Carbon-Aware Computing for Datacenters", "doi", "10.1109/TPWRS.2022.3173250"),
 ("Carbon-Aware Computing for Datacenters", "arxiv", "2106.11750"),
 ("Low Complexity Homeomorphic Projection to Ensure Neural-Network Solution Feasibility for Optimization over (Non-)Convex Set", "arxiv", "2208.07307"),
 ("Tackling Climate Change with Machine Learning", "doi", "10.1145/3485128"),
 ("Tackling Climate Change with Machine Learning", "arxiv", "1906.05433"),
 ("Task-based End-to-end Model Learning in Stochastic Optimization", "arxiv", "1703.04529"),
 ("Sustainable AI: Environmental Implications, Challenges and Opportunities", "arxiv", "2111.00364"),
 ("Orca: A Distributed Serving System for Transformer-Based Generative Models", "doi", "10.5555/3600270"),
]
out = json.load(open("verified.json"))
have = {d["query"] for d in out}
for q, kind, ident in tries:
    if q in have:
        continue
    try:
        if kind == "arxiv":
            r = requests.get(f"https://arxiv.org/abs/{ident}", headers=H, timeout=30)
            meta = lambda n: [html.unescape(m) for m in re.findall(rf'<meta name="{n}" content="([^"]*)"', r.text)]
            title = (meta("citation_title") or [""])[0]
            rec = dict(title=title, year=int((meta("citation_date") or ["0"])[0][:4]), venue=f"arXiv preprint arXiv:{ident}",
                       doi="", type="preprint", source="arxiv-abs", authors=[" ".join(reversed([p.strip() for p in a.split(",", 1)])) for a in meta("citation_author")],
                       abstract=(meta("citation_abstract") or [""])[0][:1500])
        else:
            r = requests.get(f"https://api.crossref.org/works/{ident}", headers=H, timeout=30)
            if r.status_code != 200:
                print("MISS", r.status_code, q[:60], ident); continue
            it = r.json()["message"]
            title = re.sub(r"<[^>]+>", "", (it.get("title") or [""])[0])
            rec = dict(title=title, year=(it.get("issued", {}).get("date-parts") or [[None]])[0][0],
                       venue=(it.get("container-title") or [""])[0], doi=it.get("DOI", ""), type=it.get("type"), source="crossref-doi",
                       authors=[f"{a.get('given','')} {a.get('family','')}".strip() for a in it.get("author", [])],
                       abstract=re.sub(r"<[^>]+>", "", it.get("abstract", ""))[:1500])
        s = fuzz.ratio(q.lower(), title.lower())
        if s >= 90:
            out.append(dict(query=q, score=s, **rec)); have.add(q)
            print(f"OK  {s} [{rec['source']}] {rec['year']} {title[:70]} | {rec['venue'][:30]} | {len(rec['authors'])} authors")
        else:
            print(f"REJECT {s} {q[:50]} -> '{title[:70]}'")
    except Exception as e:
        print("ERR", q[:50], e)
# reject the 2016 report matched for the 2024 query
out = [d for d in out if not (d["query"].startswith("2024 United States") and d["year"] != 2024)]
json.dump(out, open("verified.json", "w"), indent=1)
print(len(out), "verified")
