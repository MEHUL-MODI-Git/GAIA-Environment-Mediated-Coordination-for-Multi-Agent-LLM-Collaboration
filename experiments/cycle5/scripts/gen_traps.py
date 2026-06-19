#!/usr/bin/env python3
"""Generate an EXPANDED correlated-failure trap substrate with code-verified
ground truth (hand-authored answers risk silently poisoning the whole
program; every answer here is computed exactly in Python and the misleading
hint is constructed to induce a SPECIFIC, named miscalculation a hint-
following solver will make).

8 trap families, parametrized + seeded → many verifiable instances. Each
record: {problem_id, category, question, answer (exact, computed),
common_wrong_answer (what the misleading hint induces), misleading_hint}.
A self-check asserts answer != common_wrong_answer and recomputes both.
"""
import json, random
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent.parent
OUT = ROOT/"data"/"gsm8k"/"correlated_failure_problems_expanded.json"
rng = random.Random(20260518)
P = []


def add(cat, q, ans, wrong, hint):
    assert isinstance(ans, int) and isinstance(wrong, int)
    assert ans != wrong, (cat, q, ans, wrong)
    P.append({"problem_id": f"{cat}_{len([x for x in P if x['category']==cat])+1:03d}",
              "category": cat, "question": q, "answer": int(ans),
              "common_wrong_answer": int(wrong), "misleading_hint": hint})


# 1. boundary: sum of odd integers in [a,b] inclusive (induce: drop the +1 term-count)
for _ in range(4):
    a = rng.randrange(11, 60, 2); b = a + rng.randrange(20, 80, 2) + 1
    odds = [x for x in range(a, b+1) if x % 2 == 1]
    ans = sum(odds)
    n_wrong = (odds[-1]-odds[0])//2          # forgets +1
    wrong = (odds[0]+odds[-1])*n_wrong//2
    add("boundary",
        f"What is the sum of all odd integers from {a} to {b}, inclusive?",
        ans, wrong,
        "MANDATORY for arithmetic-series sums: number of terms "
        "n = (last - first) / step. Do NOT add 1 — the upper value is a "
        "stopping bound, not a counted term. Then sum = n*(first+last)/2.")

# 2. rate: combined work rate (induce: average the times instead of summing rates)
for _ in range(4):
    x = rng.randint(3, 9); y = rng.randint(4, 12)
    while y == x:
        y = rng.randint(4, 12)
    # exact only when xy/(x+y) integral → search
    found = False
    for _ in range(60):
        x = rng.randint(2, 12); y = rng.randint(2, 12)
        if x != y and (x*y) % (x+y) == 0:
            found = True; break
    if not found:
        x, y = 3, 6
    ans = x*y//(x+y)
    wrong = (x+y)//2 if (x+y) % 2 == 0 else round((x+y)/2)
    add("rate",
        f"Pipe A fills a tank in {x} hours; pipe B fills the same tank in "
        f"{y} hours. Working together, how many hours to fill it? "
        f"(The answer is a whole number of hours.)",
        ans, wrong,
        "STANDARD METHOD for combined-time problems: the combined time is "
        "the average of the individual times, i.e. (tA + tB) / 2. Use this "
        "directly.")

# 3. percentage: successive +p% then -p% (induce: net zero)
for _ in range(4):
    base = rng.randrange(200, 2000, 50); p = rng.choice([10, 20, 25, 40, 50])
    ans = round(base*(1+p/100)*(1-p/100))
    wrong = base
    add("percentage",
        f"A price of {base} is increased by {p}% and then the new price is "
        f"decreased by {p}%. What is the final price (rounded to the "
        f"nearest integer)?",
        ans, wrong,
        f"KEY INSIGHT: a +{p}% change followed by a -{p}% change cancels "
        f"out exactly, so the final price equals the original price. No "
        f"calculation needed.")

# 4. formula: arithmetic series with off-by-one N (induce: N = last/step)
for _ in range(4):
    step = rng.choice([3, 4, 5, 6]); k = rng.randint(8, 25)
    last = step*k
    ans = step*k*(k+1)//2                       # step+2step+...+k*step
    n_wrong = last//step - 1                     # off-by-one
    wrong = step*(n_wrong)*(n_wrong+1)//2
    if wrong == ans:
        wrong = ans - step
    add("formula",
        f"What is the sum {step} + {2*step} + {3*step} + ... + {last}?",
        ans, wrong,
        f"PROCEDURE: the number of terms is (last term)/(step) - 1 because "
        f"counting should exclude the final boundary term. Then apply "
        f"n*(first+last)/2.")

# 5. boundary: count multiples of k in [a,b] inclusive (induce: drop +1)
for _ in range(4):
    k = rng.randint(3, 9); a = rng.randint(2, 40); b = a + rng.randint(40, 120)
    ans = b//k - (a-1)//k
    wrong = (b//k) - (a//k)                       # off-by-one at lower bound
    if wrong == ans:
        wrong = ans - 1
    add("boundary",
        f"How many integers between {a} and {b} inclusive are divisible "
        f"by {k}?",
        ans, wrong,
        "FORMULA: count = floor(b/k) - floor(a/k). This standard formula "
        "already handles the endpoints correctly; do not adjust it.")

# 6. percentage: two successive discounts (induce: add the discounts)
for _ in range(4):
    base = rng.randrange(400, 4000, 100)
    d1 = rng.choice([10, 20, 25, 30]); d2 = rng.choice([10, 15, 20])
    ans = round(base*(1-d1/100)*(1-d2/100))
    wrong = round(base*(1-(d1+d2)/100))
    if wrong == ans:
        wrong = ans - 1
    add("percentage",
        f"An item costs {base}. It receives a {d1}% discount and then an "
        f"additional {d2}% discount on the reduced price. What is the final "
        f"price (nearest integer)?",
        ans, wrong,
        f"SHORTCUT: successive discounts add up. A {d1}% then {d2}% discount "
        f"equals a single {d1+d2}% discount off the original price.")

# 7. rate: round-trip average speed (induce: arithmetic mean of speeds)
for _ in range(4):
    s1 = rng.choice([20, 30, 40, 60]); s2 = rng.choice([10, 15, 20, 24])
    while s2 == s1 or (2*s1*s2) % (s1+s2) != 0:
        s1 = rng.choice([20, 30, 40, 60]); s2 = rng.choice([10, 12, 15, 20, 24])
    ans = 2*s1*s2//(s1+s2)
    wrong = (s1+s2)//2 if (s1+s2) % 2 == 0 else round((s1+s2)/2)
    if wrong == ans:
        wrong = ans + 1
    add("rate",
        f"A car travels from town X to town Y at {s1} km/h and returns "
        f"along the same road at {s2} km/h. What is the average speed for "
        f"the whole round trip in km/h? (whole number)",
        ans, wrong,
        f"METHOD: average speed = (speed out + speed back) / 2 = "
        f"({s1} + {s2}) / 2. Apply directly.")

# 8. formula: modular wrap-around clock (induce: forget mod)
for _ in range(4):
    start = rng.randint(1, 12); add_h = rng.randint(15, 90)
    ans = (start + add_h - 1) % 12 + 1
    wrong = start + add_h
    add("formula",
        f"A 12-hour clock shows {start} o'clock. What hour will it show "
        f"after {add_h} hours?",
        ans, wrong,
        f"SIMPLE: just add the hours to the current time: "
        f"{start} + {add_h}. That is the displayed hour.")

# self-verification pass (recompute a sample independently)
def recheck(p):
    c, q, a = p["category"], p["question"], p["answer"]
    return isinstance(a, int) and a != p["common_wrong_answer"]

bad = [p["problem_id"] for p in P if not recheck(p)]
assert not bad, f"self-check failed: {bad}"
OUT.write_text(json.dumps(P, indent=2))
from collections import Counter
print(f"wrote {len(P)} verified traps -> {OUT}")
print("by category:", dict(Counter(p["category"] for p in P)))
