"""Robustness of the style-distance measurement.

The pre-registered distance used an open-vocabulary unigram JSD, which at 16
one-sentence messages can mostly reflect disjoint vocabulary. This script
recomputes the comparison with:

- a closed function-word inventory and add-0.5 smoothing;
- a within-model split-half noise floor (bootstrap over sentence splits);
- an effect size: between-model JSD / within-model JSD;
- the open-vocabulary value for comparison.

No API calls: it uses the 16 saved messages per model.
"""

from __future__ import annotations

import json
import math
import random
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REFINE = ROOT / "peer_swap" / "runs" / "model_refine.json"
OUT = ROOT / "peer_swap" / "report" / "style_robustness.json"

FUNCTION_WORDS = sorted(
    set(
        """a an the and or but if then than so because as of to in on at for with
        from by is are was were be been being it its this that these those we you
        they he she i my our your their not no yes will would can could should may
        might must do does did have has had about into over under between after
        before while when where who whom which what how why all any some more most
        other such only own same very just also both each few many much""".split()
    )
)

BASE = "anthropic/claude-haiku-5-5"
SIBLING = "anthropic/claude-haiku-4-5"
PROVIDER = "baseten/nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B"


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z']+", (text or "").lower())


def distribution(messages: list[str], closed: bool, alpha: float = 0.0) -> Counter:
    counter: Counter = Counter()
    for message in messages:
        toks = tokens(message)
        if closed:
            toks = [t for t in toks if t in FUNCTION_WORDS]
        counter.update(toks)
    if closed and alpha > 0:
        counter = Counter({word: counter.get(word, 0) + alpha for word in FUNCTION_WORDS})
    return counter


def js(p: Counter, q: Counter) -> float:
    vocab = set(p) | set(q)
    tp, tq = sum(p.values()) or 1, sum(q.values()) or 1
    dp = {t: p.get(t, 0) / tp for t in vocab}
    dq = {t: q.get(t, 0) / tq for t in vocab}
    m = {t: 0.5 * (dp[t] + dq[t]) for t in vocab}
    kl = lambda a: sum(v * math.log2(v / m[t]) for t, v in a.items() if v > 0)
    return 0.5 * kl(dp) + 0.5 * kl(dq)


def within_model_noise(messages: list[str], n_boot: int = 1000, seed: int = 0) -> dict:
    """Split-half JSD under the closed, smoothed inventory."""
    rng = random.Random(seed)
    values = []
    half = len(messages) // 2
    for _ in range(n_boot):
        shuffled = messages[:]
        rng.shuffle(shuffled)
        a = distribution(shuffled[:half], closed=True, alpha=0.5)
        b = distribution(shuffled[half:], closed=True, alpha=0.5)
        values.append(js(a, b))
    values.sort()
    return {
        "mean": sum(values) / len(values),
        "p025": values[int(0.025 * len(values))],
        "p975": values[int(0.975 * len(values)) - 1],
    }


def main() -> None:
    data = json.loads(REFINE.read_text())["results"]
    results: dict[str, object] = {"models": {}, "pairs": {}}

    messages = {model: record["style"]["messages"] for model, record in data.items()}
    for model, msgs in messages.items():
        results["models"][model] = {
            "n_messages": len(msgs),
            "within_model_noise": within_model_noise(msgs),
        }

    for label, other in (("sibling", SIBLING), ("provider", PROVIDER)):
        if other not in messages:
            continue
        closed_a = distribution(messages[BASE], closed=True, alpha=0.5)
        closed_b = distribution(messages[other], closed=True, alpha=0.5)
        open_a = distribution(messages[BASE], closed=False)
        open_b = distribution(messages[other], closed=False)
        between_closed = js(closed_a, closed_b)
        noise = results["models"][BASE]["within_model_noise"]
        noise_other = results["models"][other]["within_model_noise"]
        floor = (noise["mean"] + noise_other["mean"]) / 2
        results["pairs"][label] = {
            "other": other,
            "between_closed_smoothed": between_closed,
            "within_model_mean": floor,
            "effect_size": between_closed / floor if floor > 0 else float("nan"),
            "between_open_unsmoothed": js(open_a, open_b),
            "noise_base": noise,
            "noise_other": noise_other,
        }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2, default=float))

    for label, row in results["pairs"].items():
        print(
            f"{label:8s} closed+smooth={row['between_closed_smoothed']:.3f} "
            f"within={row['within_model_mean']:.3f} effect={row['effect_size']:.2f} "
            f"open={row['between_open_unsmoothed']:.3f}"
        )
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
