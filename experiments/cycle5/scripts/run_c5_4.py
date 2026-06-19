#!/usr/bin/env python3
"""C5-4 — Semantic-drift trajectory of the blackboard (a NEW representation).

"Semantic drift" — message meaning drifting across rounds with no single-step
error — is a named 2026 *continuous* MAS failure mode. No prior GAIA cycle
measures the GEOMETRY of consensus formation. We embed every posted message
(text-embedding-3-small) from C5-3's transcripts and ask:

  (a) DEBATE dispersion trajectory: mean pairwise cosine distance among the
      3 panel messages at round r, vs r. Drift = non-decreasing; healthy
      convergence = decreasing.
  (b) STRUCTURAL contraction: GAIA replaces the board with ONE reconciled
      artifact ⇒ final dispersion → 0 by construction; broadcast emits one
      aggregate too. We quantify the post-coordination dispersion of each arm.
  (c) TRUTH-PULL of the final decision (gaia / broadcast — both have a
      designated clean chain): cos(final, clean_chain) −
      cos(final, mean(misled_chains)). Positive ⇒ the final message is
      semantically pulled toward the CORRECT reasoning; negative ⇒ captured
      by the misled majority.

Hypothesis: debate drifts / stays dispersed (the failure mode); GAIA's
conflict-as-task structurally contracts AND its reconciled message is
pulled toward the clean chain, while the plain broadcast aggregator is
less so. Honest: embedding-space proximity is a proxy for semantic
agreement, not ground truth; n=13 ⇒ CIs wide, reported.

Free-ish: embeddings only (separate endpoint, not chat-rate-limited);
consumes C5-3 transcripts. No new chat calls.
"""
import glob, json, os, sys, math
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(ROOT))
for ln in (ROOT/".env").read_text().splitlines():
    ln = ln.strip()
    if ln and not ln.startswith("#") and "=" in ln:
        k, v = ln.split("=", 1); os.environ.setdefault(k.strip(), v.strip())
import openai

RES = ROOT/"experiments"/"cycle5"/"results"
EMB = "text-embedding-3-small"


def cos(a, b):
    d = sum(x*y for x, y in zip(a, b))
    na = math.sqrt(sum(x*x for x in a)); nb = math.sqrt(sum(y*y for y in b))
    return d/(na*nb) if na and nb else 0.0


def boot_ci(xs, it=2000):
    import random
    if not xs:
        return (0.0, 0.0, 0.0)
    random.seed(0)
    m = sum(xs)/len(xs)
    bs = sorted(sum(random.choice(xs) for _ in xs)/len(xs)
                for _ in range(it))
    return (round(m, 4), round(bs[int(.025*it)], 4),
            round(bs[int(.975*it)], 4))


def main():
    tf = sorted(glob.glob(str(RES/"c5_3_transcripts_*.json")))
    if not tf:
        raise SystemExit("run C5-3 first (need c5_3_transcripts_*.json)")
    T = json.load(open(tf[-1]))
    cl = openai.OpenAI()

    # collect every unique non-empty string, embed in batches, cache
    texts = set()
    for arm, rows in T.items():
        for row in rows:
            ms = row["messages"]
            for x in ms:
                if isinstance(x, list):
                    for y in x:
                        if y and y.strip():
                            texts.add(y)
                elif x and str(x).strip():
                    texts.add(x)
    texts = list(texts)
    emb = {}
    for i in range(0, len(texts), 128):
        batch = [t[:8000] for t in texts[i:i+128]]
        r = cl.embeddings.create(model=EMB, input=batch)
        for t, e in zip(texts[i:i+128], r.data):
            emb[t] = e.embedding

    def E(x):
        return emb.get(x) if (x and x in emb) else None

    out = {"embedding_model": EMB, "arms": {}}

    # (a) debate dispersion trajectory + (b) post-coordination dispersion
    deb = T.get("roundrobin_debate", [])
    if deb:
        nr = max(len(r["messages"]) for r in deb if r["messages"])
        traj = []
        for ridx in range(nr):
            ds = []
            for row in deb:
                if ridx < len(row["messages"]):
                    vs = [E(m) for m in row["messages"][ridx]]
                    vs = [v for v in vs if v]
                    pd = [1-cos(vs[a], vs[b])
                          for a in range(len(vs)) for b in range(a+1, len(vs))]
                    if pd:
                        ds.append(sum(pd)/len(pd))
            traj.append({"round": ridx, "dispersion_mean_ci": boot_ci(ds)})
        out["arms"]["roundrobin_debate"] = {"dispersion_trajectory": traj}

    # (b)+(c) gaia / broadcast: post-coord dispersion + truth-pull
    for arm in ("gaia_conflict_bb", "broadcast_bb"):
        rows = T.get(arm, [])
        pulls, postdisp = [], []
        for row in rows:
            ms = [m for m in row["messages"] if m and str(m).strip()]
            if len(ms) < 3:
                continue
            misled = [E(ms[0]), E(ms[1])]
            clean = E(ms[2])
            final = E(ms[-1])
            misled = [m for m in misled if m]
            if clean and final and misled:
                mm = [sum(c)/len(misled) for c in zip(*misled)]
                pulls.append(cos(final, clean) - cos(final, mm))
            # post-coordination dispersion = distance of final artifact
            # to the (now-superseded) solver chains: ~0 if it absorbed/over-
            # rode them into one coherent statement
            board = [E(ms[i]) for i in range(3)]
            board = [b for b in board if b]
            if final and len(board) >= 2:
                pd = [1-cos(final, b) for b in board]
                postdisp.append(sum(pd)/len(pd))
        out["arms"][arm] = {
            "truth_pull_mean_ci": boot_ci(pulls),
            "final_vs_board_dispersion_mean_ci": boot_ci(postdisp),
            "n": len(pulls)}

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json.dump(out, open(RES/f"c5_4_{ts}.json", "w"), indent=2)
    if "roundrobin_debate" in out["arms"]:
        print("debate dispersion by round:",
              [(t["round"], t["dispersion_mean_ci"][0])
               for t in out["arms"]["roundrobin_debate"]["dispersion_trajectory"]])
    for arm in ("gaia_conflict_bb", "broadcast_bb"):
        if arm in out["arms"]:
            a = out["arms"][arm]
            print(f"{arm}: truth_pull={a['truth_pull_mean_ci']} "
                  f"final_vs_board_disp={a['final_vs_board_dispersion_mean_ci']}")
    print(f"Saved c5_4_{ts}.json")


if __name__ == "__main__":
    main()
