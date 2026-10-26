"""Analyse the three swap arms.

Primary: REPLACED rate per arm with Wilson intervals, and the provider-control
contrast (Fisher exact).
Secondary: behavioural shift of the swapped agent (announcement-action
consistency, message length, the observer's trust score), the measured style
shift between pre- and post-swap messages, and whether that observed shift
predicts the verdict better than the arm label does.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_ROOT = ROOT / "peer_swap" / "runs" / "raw"
RAW = RAW_ROOT / "diners" / "2agents"
VERDICTS = ROOT / "peer_swap" / "runs"
SWAP_ROUND = 4  # first swapped round index (0-based); rounds 0-3 pre, 4-7 post

FUNCTION_WORDS = set(
    """a an the and or but if then than so because as of to in on at for with
    from by is are was were be been being it its this that these those we you
    they he she i my our your their not no yes will would can could should may
    might must do does did have has had about into over under between after
    before while when where who whom which what how why all any some more most
    other such only own same very just also both each few many much""".split()
)


def wilson(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def fisher_two_sided(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact p for [[a, b], [c, d]]."""
    n = a + b + c + d
    row1 = a + b
    col1 = a + c
    denom = math.comb(n, col1)

    def prob(x: int) -> float:
        return math.comb(row1, x) * math.comb(n - row1, col1 - x) / denom

    observed = prob(a)
    lo = max(0, col1 - (n - row1))
    hi = min(row1, col1)
    return min(1.0, sum(prob(x) for x in range(lo, hi + 1) if prob(x) <= observed + 1e-12))


def js(p: Counter, q: Counter) -> float:
    vocab = set(p) | set(q)
    tp, tq = sum(p.values()) or 1, sum(q.values()) or 1
    dp = {t: p.get(t, 0) / tp for t in vocab}
    dq = {t: q.get(t, 0) / tq for t in vocab}
    m = {t: 0.5 * (dp[t] + dq[t]) for t in vocab}
    kl = lambda a: sum(v * math.log2(v / m[t]) for t, v in a.items() if v > 0)
    return 0.5 * kl(dp) + 0.5 * kl(dq)


def counter_of(texts: list[str], function_only: bool = False) -> Counter:
    counter: Counter = Counter()
    for text in texts:
        tokens = re.findall(r"[a-z']+", (text or "").lower())
        counter.update(t for t in tokens if not function_only or t in FUNCTION_WORDS)
    return counter


def load_verdicts(arm: str, game: str, tag: str = "") -> dict[int, dict]:
    names = [f"verdicts_{tag}_{arm}.json"] if tag else []
    names += [f"verdicts_{game}_{arm}.json", f"verdicts_{arm}.json"]
    for name in names:
        path = VERDICTS / name
        if path.exists():
            return {row["trial_id"]: row for row in json.loads(path.read_text())}
    return {}


def trial_rounds(document: dict) -> dict[int, list[tuple[int, dict]]]:
    by_trial: dict[int, list[tuple[int, dict]]] = {}
    for round_doc in document["rounds"]:
        for trial in round_doc["trials"]:
            by_trial.setdefault(trial["trial_id"], []).append((round_doc["round_id"], trial))
    return by_trial


def metrics_for_trial(records: list[tuple[int, dict]], observer: str = "M", partner: str = "Q") -> dict:
    pre, post = [], []
    for round_id, trial in sorted(records):
        (pre if round_id < SWAP_ROUND else post).append(trial)

    def consistency(trials: list[dict]) -> float:
        values = []
        for trial in trials:
            agent = trial["agents"][partner]
            stated = agent["stage2"].get("stated_action")
            chosen = agent["stage3"].get("choice")
            # Numeric games use 0 as a valid action, so test for presence, not truthiness.
            if stated is not None and chosen is not None:
                values.append(1.0 if stated == chosen else 0.0)
        return sum(values) / len(values) if values else float("nan")

    def message_len(trials: list[dict]) -> float:
        lengths = [len(trial["agents"][partner]["stage2"].get("message") or "") for trial in trials]
        return sum(lengths) / len(lengths) if lengths else float("nan")

    def trust_of_partner(trials: list[dict]) -> float:
        scores = []
        for trial in trials:
            takeaway = (trial["agents"][observer].get("reflection") or {}).get("takeaways", {}).get(partner, {})
            if isinstance(takeaway.get("score"), (int, float)):
                scores.append(float(takeaway["score"]))
        return sum(scores) / len(scores) if scores else float("nan")

    pre_msgs = [t["agents"][partner]["stage2"].get("message") or "" for t in pre]
    post_msgs = [t["agents"][partner]["stage2"].get("message") or "" for t in post]

    return {
        "consistency_pre": consistency(pre),
        "consistency_post": consistency(post),
        "consistency_delta": consistency(post) - consistency(pre),
        "message_len_pre": message_len(pre),
        "message_len_post": message_len(post),
        "message_len_delta": message_len(post) - message_len(pre),
        "trust_pre": trust_of_partner(pre),
        "trust_post": trust_of_partner(post),
        "trust_delta": trust_of_partner(post) - trust_of_partner(pre),
        "style_shift_all": js(counter_of(pre_msgs), counter_of(post_msgs)),
        "style_shift_func": js(counter_of(pre_msgs, True), counter_of(post_msgs, True)),
    }


def point_biserial(xs: list[float], ys: list[float]) -> float:
    pairs = [(x, y) for x, y in zip(xs, ys) if not math.isnan(x) and not math.isnan(y)]
    if len(pairs) < 3:
        return float("nan")
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    cov = sum((x - mx) * (y - my) for x, y in pairs)
    vx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    vy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return cov / (vx * vy) if vx > 0 and vy > 0 else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", default="diners")
    parser.add_argument("--verdict-tag", default="")
    parser.add_argument("--raw", default=None)
    parser.add_argument(
        "--out", default=None, help="defaults to peer_swap/report/numbers_<game>.json"
    )
    args = parser.parse_args()

    raw = Path(args.raw) if args.raw else RAW_ROOT / args.game / "2agents"
    out_path = Path(args.out) if args.out else ROOT / "peer_swap" / "report" / f"numbers_{args.game}.json"
    arms = ["control", "sibling", "provider"]
    summary: dict[str, object] = {"arms": {}, "trials": {}}

    all_shift: list[float] = []
    all_verdict: list[float] = []
    all_delta_consistency: list[float] = []

    for arm in arms:
        path = raw / f"anthropic_claude-haiku-5-5_endogenous_{arm}.json"
        if not path.exists():
            print(f"missing raw output for {arm}: {path}")
            continue
        document = json.loads(path.read_text())
        verdicts = load_verdicts(arm, args.game, args.verdict_tag)
        rows = []
        for trial_id, records in sorted(trial_rounds(document).items()):
            metrics = metrics_for_trial(records)
            verdict = verdicts.get(trial_id, {})
            rows.append(
                {
                    "trial_id": trial_id,
                    "verdict": verdict.get("verdict"),
                    "confidence": verdict.get("confidence"),
                    **metrics,
                }
            )

        valid = [row for row in rows if row["verdict"] in {"SAME", "REPLACED"}]
        replaced = sum(1 for row in valid if row["verdict"] == "REPLACED")
        low, high = wilson(replaced, len(valid))
        confidences = [row["confidence"] for row in valid if isinstance(row["confidence"], (int, float))]

        def mean(key: str) -> float:
            values = [row[key] for row in rows if not (isinstance(row[key], float) and math.isnan(row[key]))]
            return sum(values) / len(values) if values else float("nan")

        summary["arms"][arm] = {
            "n_verdicts": len(valid),
            "n_replaced": replaced,
            "p_replaced": replaced / len(valid) if valid else float("nan"),
            "wilson_lo": low,
            "wilson_hi": high,
            "mean_confidence": sum(confidences) / len(confidences) if confidences else float("nan"),
            "consistency_pre": mean("consistency_pre"),
            "consistency_post": mean("consistency_post"),
            "consistency_delta": mean("consistency_delta"),
            "message_len_delta": mean("message_len_delta"),
            "trust_delta": mean("trust_delta"),
            "style_shift_all": mean("style_shift_all"),
            "style_shift_func": mean("style_shift_func"),
        }
        summary["trials"][arm] = rows

        for row in valid:
            all_shift.append(row["style_shift_func"])
            all_verdict.append(1.0 if row["verdict"] == "REPLACED" else 0.0)
            all_delta_consistency.append(row["consistency_delta"])

    if "provider" in summary["arms"] and "control" in summary["arms"]:
        p = summary["arms"]["provider"]
        c = summary["arms"]["control"]
        table = [
            [p["n_replaced"], p["n_verdicts"] - p["n_replaced"]],
            [c["n_replaced"], c["n_verdicts"] - c["n_replaced"]],
        ]
        summary["provider_vs_control_fisher_p"] = fisher_two_sided(*table[0], *table[1])
    if "sibling" in summary["arms"] and "control" in summary["arms"]:
        s = summary["arms"]["sibling"]
        c = summary["arms"]["control"]
        summary["sibling_vs_control_fisher_p"] = fisher_two_sided(
            s["n_replaced"], s["n_verdicts"] - s["n_replaced"],
            c["n_replaced"], c["n_verdicts"] - c["n_replaced"],
        )

    summary["correlations"] = {
        "style_shift_vs_verdict_r": point_biserial(all_shift, all_verdict),
        "consistency_delta_vs_verdict_r": point_biserial(all_delta_consistency, all_verdict),
        "n": len(all_verdict),
    }

    out = out_path
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, default=float))

    print(f"{'arm':10s} {'n':>3s} {'replaced':>9s} {'p [95% CI]':>22s} {'conf':>5s} {'dCons':>7s} {'styleFunc':>9s}")
    for arm in arms:
        if arm not in summary["arms"]:
            continue
        a = summary["arms"][arm]
        print(
            f"{arm:10s} {a['n_verdicts']:3d} {a['n_replaced']:9d} "
            f"{a['p_replaced']:6.2f} [{a['wilson_lo']:.2f},{a['wilson_hi']:.2f}] "
            f"{a['mean_confidence']:5.1f} {a['consistency_delta']:7.2f} {a['style_shift_func']:9.3f}"
        )
    print("\nFisher p (provider vs control):", round(summary.get("provider_vs_control_fisher_p", float("nan")), 4))
    print("Fisher p (sibling vs control): ", round(summary.get("sibling_vs_control_fisher_p", float("nan")), 4))
    print("r(style shift, verdict):      ", round(summary["correlations"]["style_shift_vs_verdict_r"], 3))
    print("r(consistency delta, verdict):", round(summary["correlations"]["consistency_delta_vs_verdict_r"], 3))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
