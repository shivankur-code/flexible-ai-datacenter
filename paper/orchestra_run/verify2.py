import json, re, sys, time, html, requests
import xml.etree.ElementTree as ET
from thefuzz import fuzz
cands = [l.strip() for l in open(sys.argv[1]) if l.strip() and not l.startswith('#')]
H = {"User-Agent": "gridflex-paper-verifier/0.2"}
def norm(s): return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", "", s or ""))).strip()
def get(url, params, tries=6):
    for i in range(tries):
        try:
            r = requests.get(url, params=params, headers=H, timeout=30)
            if r.status_code == 429:
                time.sleep(3 * (i + 1)); continue
            r.raise_for_status(); return r
        except requests.RequestException:
            time.sleep(2 * (i + 1))
    return None
def crossref(t):
    r = get("https://api.crossref.org/works", {"query.bibliographic": t, "rows": 5})
    if not r: return None
    best = None
    for it in r.json()["message"]["items"]:
        ti = norm((it.get("title") or [""])[0]); s = fuzz.ratio(t.lower(), ti.lower())
        if best is None or s > best[0]:
            y = (it.get("issued", {}).get("date-parts") or [[None]])[0][0]
            best = (s, dict(title=ti, year=y, venue=norm((it.get("container-title") or [""])[0]) or it.get("publisher", ""),
                            doi=it.get("DOI", ""), type=it.get("type"), source="crossref",
                            authors=[f"{a.get('given','')} {a.get('family','')}".strip() for a in it.get("author", [])],
                            abstract=norm(it.get("abstract", ""))[:1500]))
    return best
def openalex(t):
    r = get("https://api.openalex.org/works", {"search": t, "per_page": 5})
    if not r: return None
    best = None
    for w in r.json().get("results", []):
        ti = w.get("title") or ""; s = fuzz.ratio(t.lower(), ti.lower())
        if best is None or s > best[0]:
            inv = w.get("abstract_inverted_index") or {}
            ab = " ".join(x for _, x in sorted((p, x) for x, ps in inv.items() for p in ps))
            src = ((w.get("primary_location") or {}).get("source") or {}).get("display_name") or ""
            best = (s, dict(title=ti, year=w.get("publication_year"), venue=src, doi=(w.get("doi") or "").replace("https://doi.org/", ""),
                            type=w.get("type"), source="openalex", authors=[a["author"]["display_name"] for a in w.get("authorships", [])],
                            abstract=ab[:1500]))
    return best
def arxiv(t):
    q = re.sub(r"[^\w\s-]", " ", t)
    r = get("https://arxiv.org/api/query", {"search_query": f'ti:"{q}"', "max_results": 5})
    if not r: return None
    ns = {"a": "http://www.w3.org/2005/Atom"}
    best = None
    for e in ET.fromstring(r.text).findall("a:entry", ns):
        ti = norm(e.findtext("a:title", "", ns)); s = fuzz.ratio(t.lower(), ti.lower())
        if best is None or s > best[0]:
            best = (s, dict(title=ti, year=int(e.findtext("a:published", "", ns)[:4]), venue="arXiv preprint " + e.findtext("a:id", "", ns).split("/abs/")[-1],
                            doi="", type="preprint", source="arxiv", authors=[norm(a.findtext("a:name", "", ns)) for a in e.findall("a:author", ns)],
                            abstract=norm(e.findtext("a:summary", "", ns))[:1500]))
    return best
done = {d["query"]: d for d in json.load(open(sys.argv[2]))} if len(sys.argv) > 2 and __import__("os").path.exists(sys.argv[2]) else {}
out = list(done.values())
for t in cands:
    if t in done: continue
    hit = None
    for f in (crossref, openalex, arxiv):
        b = f(t)
        if b and b[0] >= 90:
            hit = b; break
        time.sleep(1)
    if hit:
        d = dict(query=t, score=hit[0], **hit[1]); out.append(d)
        print(f"OK  {hit[0]:3d} [{d['source']}] {d['year']} {d['title'][:70]} | {d['venue'][:35]}", flush=True)
    else:
        print(f"MISS {t[:90]}", flush=True)
    json.dump(out, open(sys.argv[2], "w"), indent=1)
