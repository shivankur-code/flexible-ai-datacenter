# paper-orchestra run

How [`paper/paper.pdf`](../paper.pdf) was produced with Google's
[PaperOrchestra](https://github.com/google-research/paper-orchestra) pipeline. The inputs are
`paper/raw_materials/` (`idea_sparse.md`, the generated `experimental_log.md`, and `figures/` with `info.json`).

**Backend.** PaperOrchestra calls Gemini directly in its outline, literature-review (Google-Search-grounded) and
refinement agents, and no Gemini key was available in this environment. So its agents were executed **in-session by
Claude**, following PaperOrchestra's own prompts (`methods/prompts/*`) and stage order:

| Stage | PaperOrchestra agent | Output here |
|---|---|---|
| 1 | Outline agent | `outline.json` (plotting plan, intro/related-work search plan, section plan) |
| 2 | Literature-review agent | `verified.json` -> `citation_map.json`, `latex/references.bib` |
| 3 | Section-writing agent | `latex/paper.tex` (ICLR 2025 template from `templates/iclr2025`) |
| 4 | Content-refinement agent | reviewer pass: teaser map moved to the intro, floats kept before the conclusion, page limit checked |
| — | Plotting agent | not used: the figures are the experiment figures (`--use_plotting false` mode) |

**Anti-hallucination measures** (stricter than the default run):

- Every number in the paper comes from `experimental_log.md`, which `paper/build_raw_materials.py` generates
  from `results/*.csv`.
- Every reference was checked against bibliographic APIs. Crossref and OpenAlex were queried by title; arXiv abstract
  pages and Crossref were queried by identifier. A candidate was kept only if the fuzzy title match was 90 or more
  (`verify2.py`, `verify3.py`). BibTeX is generated from the API record (`build_bib.py`), never typed from memory.
  29 of 33 candidates were verified. Unverified works (Orca, homeomorphic projection, the 2024 LBNL report, POLCA)
  are not cited. One false match (a 2016 report returned for a 2024 query) was rejected by its year.
- `build_bib.py` fails loudly if any `\cite` key lacks a verified entry.

**Deliberate deviation:** PaperOrchestra's refinement prompt says "never explicitly state a limitation". The paper
keeps a Limitations paragraph, because the study's scope (one test system, one campus at a time, assumed fleet
parameters) must be stated.

**Format:** the ICLR 2025 style is used in final-copy mode (`\iclrfinalcopy`), so the author is shown, and the
running header is overridden to "Preprint". The default anonymous mode prints "Anonymous authors" and
"Under review as a conference paper at ICLR 2025", which would be inaccurate for this preprint.

Rebuild:
```bash
cd paper/orchestra_run && python build_bib.py verified.json && cd latex && pdflatex paper && bibtex paper && pdflatex paper && pdflatex paper
```
