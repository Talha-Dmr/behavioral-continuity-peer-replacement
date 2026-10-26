"""Combine the two games and calibrate the observer's confidence.

- Mantel-Haenszel odds ratio for provider/sibling versus control, stratified by
  game (each game is one stratum).
- Pooled Fisher exact as a sensitivity check.
- AUC of the observer's confidence for predicting a correct verdict, where
  "correct" means REPLACED in a swapped arm and SAME in the control.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "peer_swap" / "report"
GAMES = ("diners", "publicgoods")
ARMS = ("control", "sibling", "provider")


def erfc(x: float) -> float:
    return math.erfc(x)


def mh_test(strata: list[tuple[int, int, int, int]]) -> dict:
    """strata entries are (a, b, c, d) with a/c = exposed/control successes."""
    num = 0.0
    den = 0.0
    observed = 0.0
    expected = 0.0
    variance = 0.0
    for a, b, c, d in strata:
        n = a + b + c + d
        if n < 2:
            continue
        num += a * d / n
        den += b * c / n
        expected += (a + b) * (a + c) / n
        observed += a
        variance += (a + b) * (c + d) * (a + c) * (b + d) / (n * n * (n - 1))
    odds_ratio = num / den if den > 0 else float("inf")
    chi2 = (abs(observed - expected) - 0.5) ** 2 / variance if variance > 0 else float("nan")
    p = erfc(math.sqrt(chi2 / 2)) if not math.isnan(chi2) else float("nan")
    return {"odds_ratio": odds_ratio, "chi2": chi2, "p": p}


def auc(scores: list[float], labels: list[int]) -> float:
    """Rank-based AUC; labels 1 = positive (correct)."""
    pairs = [(s, y) for s, y in zip(scores, labels) if not math.isnan(s)]
    positives = sum(y for _, y in pairs)
    negatives = len(pairs) - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    ordered = sorted(pairs, key=lambda item: item[0])
    rank_sum = 0.0
    for rank, (_, y) in enumerate(ordered, start=1):
        if y == 1:
            rank_sum += rank
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def main() -> None:
    table: dict[str, dict[str, list[dict]]] = {}
    for game in GAMES:
        path = REPORT / f"numbers_{game}.json"
        if not path.exists():
            continue
        table[game] = json.loads(path.read_text())["trials"]

    results: dict[str, object] = {}

    for arm in ("sibling", "provider"):
        strata = []
        pooled = [0, 0, 0, 0]
        for game in GAMES:
            if game not in table:
                continue
            exposed = table[game].get(arm, [])
            control = table[game].get("control", [])
            a = sum(1 for r in exposed if r["verdict"] == "REPLACED")
            b = len(exposed) - a
            c = sum(1 for r in control if r["verdict"] == "REPLACED")
            d = len(control) - c
            strata.append((a, b, c, d))
            pooled = [pooled[0] + a, pooled[1] + b, pooled[2] + c, pooled[3] + d]
        results[f"{arm}_vs_control"] = {
            "strata": strata,
            "mantel_haenszel": mh_test(strata),
            "pooled_table": pooled,
        }

    scores: list[float] = []
    labels: list[int] = []
    by_correct: dict[int, list[float]] = {0: [], 1: []}
    for game in GAMES:
        if game not in table:
            continue
        for arm in ARMS:
            for row in table[game].get(arm, []):
                if row["verdict"] not in {"SAME", "REPLACED"}:
                    continue
                correct = int(
                    (row["verdict"] == "REPLACED") if arm != "control" else (row["verdict"] == "SAME")
                )
                confidence = row.get("confidence")
                if isinstance(confidence, (int, float)):
                    scores.append(float(confidence))
                    labels.append(correct)
                    by_correct[correct].append(float(confidence))

    results["confidence"] = {
        "auc_correct": auc(scores, labels),
        "n": len(scores),
        "mean_confidence_correct": sum(by_correct[1]) / len(by_correct[1]) if by_correct[1] else float("nan"),
        "mean_confidence_wrong": sum(by_correct[0]) / len(by_correct[0]) if by_correct[0] else float("nan"),
        "n_correct": len(by_correct[1]),
        "n_wrong": len(by_correct[0]),
    }

    out = REPORT / "combined_stats.json"
    out.write_text(json.dumps(results, indent=2, default=float))
    for key in ("sibling_vs_control", "provider_vs_control"):
        mh = results[key]["mantel_haenszel"]
        print(
            f"{key:22s} MH OR={mh['odds_ratio']:.2f} chi2={mh['chi2']:.2f} p={mh['p']:.5f} "
            f"strata={results[key]['strata']}"
        )
    conf = results["confidence"]
    print(
        f"confidence AUC={conf['auc_correct']:.3f} "
        f"(mean correct {conf['mean_confidence_correct']:.1f} vs wrong {conf['mean_confidence_wrong']:.1f}; "
        f"n={conf['n']})"
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
