#!/usr/bin/env python3
"""C3-5 — Semantic reasoning-chain analysis (deepens W10's char-length finding).

W10 found failed episodes have LONGER reasoning (4200 vs 2824 chars) — but
length is a crude proxy. C3-5 embeds the actual reasoning content
(text-embedding-3-small) from E3 state dumps and asks:
  (1) Do roles reason in semantically DISTINCT ways? → mean pairwise cosine
      between role-centroids + silhouette of role labels in embedding space.
      (Tests if the reconciler's reasoning differs in *kind*, not just
      length, from solvers — strengthens the E3 mechanism story.)
  (2) Is semantic *spread* within an episode predictive of success?
      (intra-episode dispersion vs passed) — a content signature, not length.

Cheap (~$0.10 embeddings). Honest: cosine/centroid geometry is a proxy for
"reasoning kind"; reported as such.
"""
import json, glob, math, os, sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(ROOT))
for ln in (ROOT/".env").read_text().splitlines():
    ln = ln.strip()
    if ln and not ln.startswith("#") and "=" in ln:
        k, v = ln.split("=", 1); os.environ.setdefault(k.strip(), v.strip())
import openai

OUT = ROOT/"experiments"/"cycle3"
ROLE = {"math_solution": "solver", "reconciled_solution": "reconciler",
        "aggregator_verdict": "aggregator", "partial_deduction": "expert",
        "proposed_solution": "synthesizer", "trust_audit": "auditor"}


def cos(a, b):
    d = sum(x*y for x, y in zip(a, b))
    na = math.sqrt(sum(x*x for x in a)); nb = math.sqrt(sum(y*y for y in b))
    return d/(na*nb) if na and nb else 0.0


def main():
    # collect (role, text, passed, episode) from E3 + E9 gaia dumps (rich chains)
    items = []
    for sf in glob.glob(str(ROOT/"experiments/correlated_failure/logs/**/*.state.json"),
                        recursive=True):
        try:
            d = json.load(open(sf))
        except Exception:
            continue
        passed = bool(d.get("extra", {}).get("passed"))
        eid = d.get("episode_id", Path(sf).stem)
        for a in d.get("artifacts", {}).values():
            r = ROLE.get(a.get("metadata", {}).get("subtype"))
            c = (a.get("content") or "").strip()
            if r and len(c) > 40:
                items.append({"role": r, "text": c[:2500], "passed": passed,
                              "ep": eid})
    if len(items) < 6:
        print("insufficient chains"); return
    cl = openai.OpenAI()
    embs = []
    B = 64
    for i in range(0, len(items), B):
        chunk = [it["text"] for it in items[i:i+B]]
        r = cl.embeddings.create(model="text-embedding-3-small", input=chunk)
        embs += [e.embedding for e in r.data]
    for it, e in zip(items, embs):
        it["e"] = e

    # (1) role-centroid geometry
    byrole = defaultdict(list)
    for it in items:
        byrole[it["role"]].append(it["e"])
    cents = {r: [sum(c)/len(v) for c in zip(*v)] for r, v in byrole.items() if v}
    roles = sorted(cents)
    pair = []
    for i in range(len(roles)):
        for j in range(i+1, len(roles)):
            pair.append((f"{roles[i]}~{roles[j]}",
                         round(cos(cents[roles[i]], cents[roles[j]]), 3)))
    # silhouette-ish: mean in-role cohesion vs nearest other-role centroid
    sil = {}
    for r in roles:
        coh = sum(cos(e, cents[r]) for e in byrole[r])/len(byrole[r])
        others = max(cos(cents[r], cents[o]) for o in roles if o != r) if len(roles) > 1 else 0
        sil[r] = round(coh - others, 3)

    # (2) intra-episode semantic spread vs success
    byep = defaultdict(list)
    for it in items:
        byep[it["ep"]].append(it)
    sp_pass, sp_fail = [], []
    for ep, its in byep.items():
        if len(its) < 2:
            continue
        cen = [sum(c)/len(its) for c in zip(*[x["e"] for x in its])]
        spread = sum(1-cos(x["e"], cen) for x in its)/len(its)
        (sp_pass if its[0]["passed"] else sp_fail).append(spread)

    def m(x): return round(sum(x)/len(x), 4) if x else None
    L = ["# C3-5 — Semantic reasoning-chain analysis (E3, n_chains="
         f"{len(items)})", "",
         "## (1) Do roles reason in a semantically distinct *kind*?",
         "Role-centroid pairwise cosine (1.0=identical kind, lower=distinct):"]
    for nm, v in pair:
        L.append(f"- {nm}: {v}")
    L += ["", "Role separation score (in-role cohesion − nearest other "
          "centroid; higher = more distinct reasoning kind):"]
    for r in roles:
        L.append(f"- {r}: {sil[r]}  (n={len(byrole[r])})")
    L += ["", "## (2) Intra-episode semantic spread vs success",
          f"- mean spread | PASSED episodes: {m(sp_pass)} (n={len(sp_pass)})",
          f"- mean spread | FAILED episodes: {m(sp_fail)} (n={len(sp_fail)})",
          "",
          "## Reading",
          "- If reconciler↔solver centroid cosine is materially below "
          "solver↔solver, the reconciler reasons in a *different kind*, not "
          "merely longer — upgrading W10's length proxy to a semantic claim "
          "and reinforcing the E3 mechanism (the reconciler does a "
          "qualitatively different audit).",
          "- Spread vs success links *content diversity* (not char length) to "
          "outcome. Honest: embedding-geometry is a proxy for 'reasoning "
          "kind'; ~$0.1, fully reproducible (text-embedding-3-small)."]
    (OUT/"figures"/"c3_5_semantics.md").write_text("\n".join(L))
    json.dump({"pairwise": pair, "separation": sil,
               "spread_pass": m(sp_pass), "spread_fail": m(sp_fail),
               "n_chains": len(items)},
              open(OUT/"results"/"c3_5_semantics.json", "w"), indent=2)
    print("\n".join(L))


if __name__ == "__main__":
    main()
