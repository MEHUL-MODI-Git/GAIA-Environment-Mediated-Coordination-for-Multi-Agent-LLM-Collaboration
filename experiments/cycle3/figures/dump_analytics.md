# C3-2/C3-3/C3-6 — Collective properties, traceability, conflict dynamics (476 dumps, free)

| exp | n | div-of-labour | inst-memory | reconstructable | provenance-ok | conflict→resolved | mean artifacts | mean localizability(fail) |
|---|---|---|---|---|---|---|---|---|
| E3 | 65 | 0.865 | 0.187 | 100% | 100% | 52→52 | 5.0 | 2.8 |
| E9 | 80 | 0.894 | 0.49 | 100% | 100% | 1→1 | 6.525 | None |
| E4 | 80 | 0.924 | 0.476 | 100% | 100% | 6→6 | 6.15 | None |
| E8 | 80 | 0.842 | 0.5 | 100% | 100% | 0→0 | 3.75 | None |
| other | 167 | 0.911 | 0.301 | 100% | 100% | 73→73 | 5.527 | 1.0 |

## Reading
- **Traceability (C3-3):** reconstructable + provenance-ok rates near 1.0 across experiments ⇒ GAIA episodes are *auditable by construction* — directly addressing the SOTA finding that 86-89% of agent pilots fail from missing traceability. Mean localizability = how many artifacts an auditor inspects to reach the failure-causing one (small ⇒ cheap root-cause).
- **Division of labour (C3-2):** normalized role-action entropy; higher = work genuinely shared across roles (not one agent doing everything). Compare GAIA pipelines vs degenerate cases.
- **Institutional memory (C3-2):** downstream artifacts reuse a substantial token-fraction first introduced upstream ⇒ the board carries information forward (not per-agent restart). Lexical proxy, stated as such.
- **Conflict dynamics (C3-6):** conflict→resolved counts + artifact spread; complements E5/W9.

Honest: institutional-memory & localizability are lexical/structural proxies (no semantics); reported as proxies. Free re-analysis of existing dumps — no new runs, fully reproducible.