"""Exploratory model: do observed style and behavioural change predict the verdict?

Ridge logistic regression with two standardized predictors (observed function-word
style shift and announcement-action consistency delta), pooled over games and arms.
Arms are deliberately left out: the provider arm has no false negatives, so arm
dummies would separate and the coefficients would diverge. Bootstrap percentile
intervals are reported; this is a secondary, exploratory analysis.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "peer_swap" / "report"
GAMES = ("diners", "publicgoods")
LAMBDA = 1.0
N_BOOT = 2000


def load_rows() -> list[dict]:
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
                        "y": 1.0 if row["verdict"] == "REPLACED" else 0.0,
                        "style": row["style_shift_func"],
                        "consistency": row["consistency_delta"],
                    }
                )
    return rows


def design(rows: list[dict], style_mu: float, style_sd: float, cons_mu: float, cons_sd: float) -> np.ndarray:
    x = np.ones((len(rows), 3))
    x[:, 1] = (np.array([r["style"] for r in rows]) - style_mu) / style_sd
    x[:, 2] = (np.array([r["consistency"] for r in rows]) - cons_mu) / cons_sd
    return x


def fit_logistic(x: np.ndarray, y: np.ndarray, lam: float = LAMBDA) -> np.ndarray:
    beta = np.zeros(x.shape[1])
    penalty = np.eye(x.shape[1])
    penalty[0, 0] = 0.0  # do not penalise the intercept
    for _ in range(100):
        eta = x @ beta
        p = 1.0 / (1.0 + np.exp(-eta))
        w = np.clip(p * (1.0 - p), 1e-9, None)
        grad = x.T @ (y - p) - lam * (penalty @ beta)
        hess = x.T @ (x * w[:, None]) + lam * penalty
        step = np.linalg.solve(hess, grad)
        beta += step
        if np.max(np.abs(step)) < 1e-8:
            break
    return beta


def odds(beta: np.ndarray) -> list[float]:
    return [math.exp(float(b)) for b in beta[1:]]


def main() -> None:
    rows = load_rows()
    y = np.array([r["y"] for r in rows])
    style = np.array([r["style"] for r in rows])
    consistency = np.array([r["consistency"] for r in rows])
    style_mu, style_sd = float(style.mean()), float(style.std(ddof=1))
    cons_mu, cons_sd = float(consistency.mean()), float(consistency.std(ddof=1))

    x = design(rows, style_mu, style_sd, cons_mu, cons_sd)
    beta = fit_logistic(x, y)
    point = odds(beta)

    rng = random.Random(0)
    boot = []
    for _ in range(N_BOOT):
        idx = [rng.randrange(len(rows)) for _ in rows]
        xb = x[idx]
        yb = y[idx]
        if yb.min() == yb.max():
            continue
        boot.append(odds(fit_logistic(xb, yb)))
    boot_arr = np.array(boot)
    lo = np.percentile(boot_arr, 2.5, axis=0)
    hi = np.percentile(boot_arr, 97.5, axis=0)

    # Univariate descriptive comparison
    replaced = [r for r in rows if r["y"] == 1]
    same = [r for r in rows if r["y"] == 0]

    result = {
        "n": len(rows),
        "n_replaced": len(replaced),
        "n_same": len(same),
        "standardization": {"style_mu": style_mu, "style_sd": style_sd, "consistency_mu": cons_mu, "consistency_sd": cons_sd},
        "ridge_lambda": LAMBDA,
        "odds_ratio": {"style": point[0], "consistency_delta": point[1]},
        "odds_ratio_ci": {
            "style": [float(lo[0]), float(hi[0])],
            "consistency_delta": [float(lo[1]), float(hi[1])],
        },
        "means": {
            "style_replaced": sum(r["style"] for r in replaced) / len(replaced),
            "style_same": sum(r["style"] for r in same) / len(same),
            "consistency_replaced": sum(r["consistency"] for r in replaced) / len(replaced),
            "consistency_same": sum(r["consistency"] for r in same) / len(same),
        },
    }
    out = REPORT / "predict_verdict.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"n={len(rows)} (REPLACED {len(replaced)}, SAME {len(same)})")
    print(f"style OR per SD       = {point[0]:.2f}  [{lo[0]:.2f}, {hi[0]:.2f}]")
    print(f"consistency OR per SD = {point[1]:.2f}  [{lo[1]:.2f}, {hi[1]:.2f}]")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
