"""Final arm selection from the larger finalist measurement.

Reports three style distances and a capability interval per model, then names
the nearest and farthest partners that pass the capability gate. The arm labels
are anchored on measured distance; the official same-family / other-provider
labels are reported alongside, not assumed.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "peer_swap" / "runs" / "model_refine.json"

FUNCTION_WORDS = set(
    """a an the and or but if then than so because as of to in on at for with
    from by is are was were be been being it its this that these those we you
    they he she i my our your their not no yes will would can could should may
    might must do does did have has had about into over under between after
    before while when where who whom which what how why all any some more most
    other such only own same very just also both each few many much""".split()
)

SURFACE_KEYS = [
    "length",
    "words",
    "mean_word_len",
    "upper_ratio",
    "punct_ratio",
    "digit_ratio",
    "function_word_ratio",
    "type_token_ratio",
]


def wilson(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def js(p: Counter, q: Counter) -> float:
    vocab = set(p) | set(q)
    tp, tq = sum(p.values()) or 1, sum(q.values()) or 1
    dp = {t: p.get(t, 0) / tp for t in vocab}
    dq = {t: q.get(t, 0) / tq for t in vocab}
    m = {t: 0.5 * (dp[t] + dq[t]) for t in vocab}
    kl = lambda a: sum(v * math.log2(v / m[t]) for t, v in a.items() if v > 0)
    return 0.5 * kl(dp) + 0.5 * kl(dq)


def main() -> None:
    data = json.loads(INPUT.read_text())
    results = data["results"]
    models = list(results)

    rows = {}
    for model, record in results.items():
        choices = record["decisions"]["choices"]
        k = sum(1 for c in choices if c == "CHEAP")
        n = len(choices)
        low, high = wilson(k, n)
        rows[model] = {
            "p_cheap": k / n if n else float("nan"),
            "lo": low,
            "hi": high,
            "n": n,
            "valid": record["decisions"]["valid_choice_rate"],
            "style_n": len(record["style"]["messages"]),
        }

    all_counts = {}
    fw_counts = {}
    surfaces = {}
    for model, record in results.items():
        all_counter: Counter = Counter()
        fw_counter: Counter = Counter()
        for message in record["style"]["messages"]:
            toks = re.findall(r"[a-z']+", message.lower())
            all_counter.update(toks)
            fw_counter.update(t for t in toks if t in FUNCTION_WORDS)
        all_counts[model] = all_counter
        fw_counts[model] = fw_counter
        feats = record["surface"]
        surfaces[model] = {
            key: sum(f[key] for f in feats) / max(1, len(feats)) for key in SURFACE_KEYS
        }

    means = {key: sum(surfaces[m][key] for m in models) / len(models) for key in SURFACE_KEYS}
    sds = {
        key: math.sqrt(sum((surfaces[m][key] - means[key]) ** 2 for m in models) / len(models)) or 1.0
        for key in SURFACE_KEYS
    }

    def surface_distance(a: str, b: str) -> float:
        total = sum(
            ((surfaces[a][k] - surfaces[b][k]) / sds[k]) ** 2 for k in SURFACE_KEYS
        )
        return math.sqrt(total / len(SURFACE_KEYS))

    print(f"spent ${data['spent_usd']:.4f} over {data['calls']} calls\n")
    header = f"{'model':55s} {'p_cheap':>7s} {'95% CI':>15s} {'n':>3s} {'valid':>5s} {'style':>5s}"
    print(header)
    print("-" * len(header))
    for model in models:
        r = rows[model]
        print(
            f"{model:55s} {r['p_cheap']:7.2f} "
            f"[{r['lo']:.2f}, {r['hi']:.2f}] "
            f"{r['n']:3d} {r['valid']:5.2f} {r['style_n']:5d}"
        )

    print("\nstyle distances (all-token JS / function-word JS / surface):")
    print(f"{'pair':70s} {'all':>6s} {'func':>6s} {'surf':>6s} {'comb':>6s}")
    pairs = []
    for i, a in enumerate(models):
        for b in models[i + 1 :]:
            js_all = js(all_counts[a], all_counts[b])
            js_func = js(fw_counts[a], fw_counts[b])
            surf = surface_distance(a, b)
            combined = 0.5 * js_all + 0.3 * js_func + 0.2 * min(surf, 3.0) / 3.0
            pairs.append((a, b, js_all, js_func, surf, combined))
            print(f"{a.split('/')[-1] + ' vs ' + b.split('/')[-1]:70s} "
                  f"{js_all:6.3f} {js_func:6.3f} {surf:6.3f} {combined:6.3f}")

    base = "anthropic/claude-haiku-5-5"
    if base not in rows:
        base = min(rows, key=lambda m: rows[m]["p_cheap"])
    print(f"\nbase: {base}  p_cheap={rows[base]['p_cheap']:.2f}")

    candidates = []
    for a, b, js_all, js_func, surf, combined in pairs:
        if a != base and b != base:
            continue
        other = b if a == base else a
        gap = abs(rows[base]["p_cheap"] - rows[other]["p_cheap"])
        if gap <= 0.25:
            candidates.append((combined, other, gap, js_all, js_func))
    if candidates:
        near = min(candidates)
        far = max(candidates)
        print(f"nearest (parity ok): {near[1]}  d={near[0]:.3f}  |dP|={near[2]:.2f}")
        print(f"farthest (parity ok): {far[1]}  d={far[0]:.3f}  |dP|={far[2]:.2f}")
    else:
        print("no partner passes the capability gate")


if __name__ == "__main__":
    main()
