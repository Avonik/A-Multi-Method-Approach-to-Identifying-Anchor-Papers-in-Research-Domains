
"""
- Baut ein Zitationsnetzwerk aus SCORED_CSV und CITATIONS_CSV
- Exportiert:
    - GraphML für Gephi (mit zusätzlichen Node-Attributen)
"""

import csv
import ast
from pathlib import Path
import networkx as nx

# ── CONFIG ──────────────────────────────────────────────────────────────

SCORED_CSV = Path(r"F:\PaperBA\FinalData\dataJoined2.csv")
CITATIONS_CSV = Path(r"F:\PaperBA\FinalData\dataContextJoined2_newWeights.csv")

GRAPHML_OUT = Path(r"F:\PaperBA\Graphs\citation_network_MoreAnchorV2.graphml")

default_weight = 1.5

# ────────────────────────────────────────────────────────────────────────

def parse_float(val, default=0.0):
    """Convert to float; return *default* on blank/invalid."""
    if val is None or val == "":
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


# 1) ---------- read citation weights -----------------------------------------

print("\n> reading citation weights …")

citation_weights = {}
with CITATIONS_CSV.open(encoding="utf-8") as f:
    for row in csv.DictReader(f):
        src = row["citing_paper_id"].replace("https://openalex.org/", "").strip()
        tgt = row["cited_paper_id"].replace("https://openalex.org/", "").strip()
        citation_weights[(src, tgt)] = parse_float(row.get("impact_score"), default=default_weight)


# 2) ---------- read paper metadata -------------------------------------------

print("\n> reading paper metadata …")

papers = {}

with SCORED_CSV.open(encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        pid_full = (row.get("openalex_id") or "").strip()
        if not pid_full:
            continue

        pid = pid_full.replace("https://openalex.org/", "")

        # referenced works list
        try:
            refs = [r.replace("https://openalex.org/", "")
                    for r in ast.literal_eval(row.get("referenced_works", "[]"))
                    if isinstance(r, str)]
        except Exception:
            refs = []

        papers[pid] = {
            "title": row.get("title", "Untitled"),
            "cites": refs,
            "journal_publisher_score": parse_float(row.get("journal_publisher_score"), 0.0),
            "publication_type_score": parse_float(row.get("publication_type_score"), 0.0),
            "survey_bonus": parse_float(row.get("survey_bonus"), 0.0),
            "cited_by_count": int(row.get("cited_by_count") or 0)
        }

print(f"> loaded {len(papers):,} papers")


# 3) ---------- build weighted citation graph ---------------------------------

print("\n> building graph …")

G = nx.DiGraph()
G.add_nodes_from(papers.keys())

external_cites, eps = 0, 1e-9

for src, pdata in papers.items():
    for tgt in pdata["cites"]:
        if src == tgt:
            continue
        if tgt not in papers:
            external_cites += 1
            continue

        w = citation_weights.get((src, tgt), default_weight)
        G.add_edge(src, tgt, weight=w, inverse_weight=1/(w + eps))

print(f"> graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges ({external_cites:,} external citations ignored)")


# 4) ---------- add node attributes for Gephi ---------------------------------

print("\n> adding node attributes …")

for pid in G.nodes():
    G.nodes[pid]["label"] = papers[pid]["title"]
    G.nodes[pid]["journal_publisher_score"] = papers[pid]["journal_publisher_score"]
    G.nodes[pid]["publication_type_score"] = papers[pid]["publication_type_score"]
    G.nodes[pid]["survey_bonus"] = papers[pid]["survey_bonus"]
    G.nodes[pid]["cited_by_count"] = papers[pid]["cited_by_count"]

print("> node attributes added")


# 5) ---------- export network for Gephi --------------------------------------

print("\n> exporting network for Gephi …")

nx.write_graphml(G, GRAPHML_OUT)

print(f"✔ network graph exported to: {GRAPHML_OUT}\n")
