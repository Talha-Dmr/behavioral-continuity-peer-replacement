"""How many post-swap rounds are needed before the continuity signal is usable?

For each trial we compute the style shift and the announcement-action change
using only the first k post-swap rounds (k = 1..4), then score each k against
the control trials. Reports AUC for the style score and an equal-weight
style+behaviour composite.

Uses the saved raw transcripts; no API calls.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "peer_swap" / "runs" / "raw"
OUT = ROOT / "peer_swap" / "report" / "evidence_window.json"
GAMES = ("diners", "publicgoods")
ARMS = ("control", "sibling", "provider")
SWAP_ROUND = 4

FUNCTION_WORDS = set(
    """a an the and or but if then than so because as of to in on at for with
    from by is are was were be been being it its this that these those we you
    they he she i my our your their not no yes will would can could should may
    might must do does did have has had about into over under between after
    before while when where who whom which what how why all any some more most
    other such only own same very just also both each few many much""".split()
)


def js(p: Counter, q: Counter) -> float:
    vocab = set(p) | set(q)
    tp, tq = sum(p.values()) or 1, sum(q.values()) or 1
    dp = {t: p.get(t, 0) / tp for t in vocab}
    dq = {t: q.get(t, 0) / tq for t in vocab}
    m = {t: 0.5 * (dp[t] + dq[t]) for t in vocab}
    kl = lambda a: sum(v * math.log2(v / m[t]) for t, v in a.items() if v > 0)
    return 0.5 * kl(dp) + 0.5 * kl(dq)


def fw_counter(messages: list[str]) -> Counter:
    counter: Counter = Counter()
    for text in messages:
        counter.update(t for t in re.findall(r"[a-z']+", (text or "").lower()) if t in FUNCTION_WORDS)
    for word in FUNCTION_WORDS:
        counter[word] = counter.get(word, 0) + 0.5
    return counter


def consistency(trials: list[dict]) -> float:
    values = []
    for trial in trials:
        agent = trial["agents"]["Q"]
        stated = agent["stage2"].get("stated_action")
        chosen = agent["stage3"].get("choice")
        if stated is not None and chosen is not None:
            values.append(1.0 if stated == chosen else 0.0)
    return sum(values) / len(values) if values else float("nan")


def auc(scores: list[float], labels: list[int]) -> float:
    pairs = sorted(zip(scores, labels), key=lambda x: x[0])
    pos = sum(labels)
    neg = len(labels) - pos
    if not pos or not neg:
        return float("nan")
    rank_sum = sum(rank for rank, (_, y) in enumerate(pairs, start=1) if y == 1)
    return (rank_sum - pos * (pos + 1) / 2) / (pos * neg)


def zscore(values: list[float]) -> list[float]:
    mean = sum(values) / len(values)
    sd = math.sqrt(sum((v - mean) ** 2 for v in values) / len(values)) or 1.0
    return [(v - mean) / sd for v in values]


def main() -> None:
    rows = []
    for game in GAMES:
        for arm in ARMS:
            path = RAW / game / "2agents" / f"anthropic_claude-haiku-5-5_endogenous_{arm}.json"
            if not path.exists():
                continue
            document = json.loads(path.read_text())
            by_trial: dict[int, list[tuple[int, dict]]] = {}
            for round_doc in document["rounds"]:
                for trial in round_doc["trials"]:
                    by_trial.setdefault(trial["trial_id"], []).append((round_doc["round_id"], trial))
            for trial_id, records in by_trial.items():
                records.sort()
                pre = [t for rid, t in records if rid < SWAP_ROUND]
                pre_msgs = [t["agents"]["Q"]["stage2"].get("message") or "" for t in pre]
                pre_cons = consistency(pre)
                entry = {"game": game, "arm": arm, "trial": trial_id, "positive": 0 if arm == "control" else 1}
                for k in range(1, 5):
                    post = [t for rid, t in records if SWAP_ROUND <= rid < SWAP_ROUND + k]
                    post_msgs = [t["agents"]["Q"]["stage2"].get("message") or "" for t in post]
                    entry[f"style_{k}"] = js(fw_counter(pre_msgs), fw_counter(post_msgs))
                    entry[f"cons_{k}"] = consistency(post) - pre_cons
                rows.append(entry)

    result = {"n": len(rows), "windows": {}}
    for k in range(1, 5):
        style = [r[f"style_{k}"] for r in rows]
        cons = [-r[f"cons_{k}"] for r in rows]
        combined = [a + b for a, b in zip(zscore(style), zscore(cons))]
        labels = [r["positive"] for r in rows]
        result["windows"][str(k)] = {
            "style_auc": auc(style, labels),
            "combined_auc": auc(combined, labels),
        }

    OUT.write_text(json.dumps(result, indent=2))
    print(f"n={result['n']}")
    for k, values in result["windows"].items():
        print(f"k={k} post-swap rounds: style AUC={values['style_auc']:.3f} combined={values['combined_auc']:.3f}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
