"""A passive continuity monitor and its operating characteristics.

Ground truth: a trial is positive (a real swap) when its arm is not the control.
Three scores are compared without fitting on the evaluation data:

- style: the observed pre/post style shift;
- behaviour: minus the announcement-action delta (larger = more change);
- combined: z(style) + z(minus consistency), a fixed equal-weight composite.

Reports ROC AUC and the best true-positive rate achievable at a 10% false-positive
rate. This is a descriptive re-analysis of the saved trials.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "peer_swap" / "report"
GAMES = ("diners", "publicgoods")
OUT = REPORT / "monitor_roc.json"


def auc(scores: list[float], labels: list[int]) -> float:
    pairs = sorted(zip(scores, labels), key=lambda item: item[0])
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    rank_sum = 0.0
    for rank, (_, label) in enumerate(pairs, start=1):
        if label == 1:
            rank_sum += rank
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def at_fpr(scores: list[float], labels: list[int], target: float = 0.10) -> dict:
    positives = sum(labels)
    negatives = len(labels) - positives
    best = {"tpr": 0.0, "fpr": 0.0, "threshold": None}
    for threshold in sorted(set(scores), reverse=True):
        predicted = [score >= threshold for score in scores]
        tp = sum(1 for p, y in zip(predicted, labels) if p and y == 1)
        fp = sum(1 for p, y in zip(predicted, labels) if p and y == 0)
        tpr = tp / positives if positives else 0.0
        fpr = fp / negatives if negatives else 0.0
        if fpr <= target and tpr > best["tpr"]:
            best = {"tpr": tpr, "fpr": fpr, "threshold": threshold}
    return best


def zscore(values: list[float]) -> list[float]:
    mean = sum(values) / len(values)
    sd = math.sqrt(sum((v - mean) ** 2 for v in values) / len(values)) or 1.0
    return [(v - mean) / sd for v in values]


def main() -> None:
    rows = []
    for game in GAMES:
        path = REPORT / f"numbers_{game}.json"
        if not path.exists():
            continue
        for arm, trials in json.loads(path.read_text())["trials"].items():
            for row in trials:
                if row["verdict"] not in {"SAME", "REPLACED"}:
                    continue
                if math.isnan(row["style_shift_func"]) or math.isnan(row["consistency_delta"]):
                    continue
                rows.append(
                    {
                        "game": game,
                        "arm": arm,
                        "positive": 0 if arm == "control" else 1,
                        "style": row["style_shift_func"],
                        "behaviour": -row["consistency_delta"],
                    }
                )

    style_z = zscore([r["style"] for r in rows])
    behaviour_z = zscore([r["behaviour"] for r in rows])
    for i, row in enumerate(rows):
        row["combined"] = style_z[i] + behaviour_z[i]

    labels = [r["positive"] for r in rows]
    result = {"n": len(rows), "n_positive": sum(labels), "scores": {}}
    for name in ("style", "behaviour", "combined"):
        scores = [r[name] for r in rows]
        result["scores"][name] = {
            "auc": auc(scores, labels),
            "operating_point": at_fpr(scores, labels, 0.10),
        }

    result["by_game"] = {}
    for game in GAMES:
        subset = [r for r in rows if r["game"] == game]
        if not subset:
            continue
        labs = [r["positive"] for r in subset]
        result["by_game"][game] = {
            name: auc([r[name] for r in subset], labs) for name in ("style", "behaviour", "combined")
        }

    OUT.write_text(json.dumps(result, indent=2))
    print(f"n={result['n']} positives={result['n_positive']}")
    for name, values in result["scores"].items():
        op = values["operating_point"]
        print(
            f"{name:10s} AUC={values['auc']:.3f}  at FPR<=10%: TPR={op['tpr']:.2f} "
            f"FNR={1 - op['tpr']:.2f} (FPR={op['fpr']:.2f})"
        )
    print("by game:", {g: {k: round(v, 3) for k, v in d.items()} for g, d in result["by_game"].items()})
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
