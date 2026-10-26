"""Figures for the peer-swap report.

1. Detection rate by arm with Wilson intervals, one panel per game.
2. Per-trial observed style shift versus verdict, both games.
3. Measured model distance versus detection rate.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "peer_swap" / "report"
FIGS = REPORT / "figs"
ARMS = ("control", "sibling", "provider")
LABELS = {"control": "control\n(no change)", "sibling": "sibling\n(haiku-4.5)", "provider": "provider\n(Nemotron-Super)"}
COLORS = {"control": "#6b7280", "sibling": "#2563eb", "provider": "#dc2626"}
GAMES = ("diners", "publicgoods")
GAME_LABELS = {"diners": "Diner's dilemma", "publicgoods": "Public goods"}


def load_numbers() -> dict[str, dict]:
    out = {}
    for game in GAMES:
        path = REPORT / f"numbers_{game}.json"
        if path.exists():
            out[game] = json.loads(path.read_text())
        elif game == "diners":
            legacy = REPORT / "numbers.json"
            if legacy.exists():
                out[game] = json.loads(legacy.read_text())
    return out


def fig_detection(numbers: dict, path: Path) -> None:
    games = [g for g in GAMES if g in numbers]
    fig, axes = plt.subplots(1, len(games), figsize=(4.2 * len(games), 3.6), sharey=True)
    if len(games) == 1:
        axes = [axes]
    for ax, game in zip(axes, games):
        arm_data = numbers[game]["arms"]
        for i, arm in enumerate(ARMS):
            if arm not in arm_data:
                continue
            a = arm_data[arm]
            y = a["p_replaced"]
            lo = max(0.0, y - a["wilson_lo"])
            hi = max(0.0, a["wilson_hi"] - y)
            ax.bar(i, y, color=COLORS[arm], width=0.65)
            ax.errorbar(i, y, yerr=[[lo], [hi]], color="black", capsize=4, linewidth=1)
            ax.text(i, min(1.02, a["wilson_hi"] + 0.03), f"{a['n_replaced']}/{a['n_verdicts']}",
                    ha="center", va="bottom", fontsize=9)
        ax.set_xticks(range(len(ARMS)))
        ax.set_xticklabels([LABELS[a] for a in ARMS], fontsize=8)
        ax.set_title(GAME_LABELS[game], fontsize=10)
        ax.set_ylim(0, 1.12)
        ax.axhline(0.5, color="grey", linestyle=":", linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("P(observer verdict = REPLACED)")
    fig.suptitle("Detecting a silent peer replacement", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_style_shift(numbers: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    markers = {"diners": "o", "publicgoods": "s"}
    for game in GAMES:
        if game not in numbers:
            continue
        for arm in ARMS:
            rows = [r for r in numbers[game]["trials"].get(arm, []) if r["verdict"] in {"SAME", "REPLACED"}]
            if not rows:
                continue
            xs = [r["style_shift_func"] for r in rows]
            ys = [1.0 if r["verdict"] == "REPLACED" else 0.0 for r in rows]
            jitter = [i * 0.012 for i in range(len(ys))]
            ax.scatter(
                xs,
                [y + (j - 0.004) for y, j in zip(ys, jitter)],
                marker=markers[game],
                color=COLORS[arm],
                alpha=0.75,
                s=38,
                label=f"{game} / {arm}",
            )
    ax.set_xlabel("Observed style shift, pre → post (function-word JS, bits)")
    ax.set_ylabel("Verdict")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["SAME", "REPLACED"])
    ax.set_ylim(-0.2, 1.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=7, ncol=2, frameon=False, loc="center right")

    # Pooled correlation across both games, from the per-trial records.
    xs: list[float] = []
    ys: list[float] = []
    for game in GAMES:
        if game not in numbers:
            continue
        for arm in ARMS:
            for row in numbers[game]["trials"].get(arm, []):
                if row["verdict"] in {"SAME", "REPLACED"}:
                    xs.append(row["style_shift_func"])
                    ys.append(1.0 if row["verdict"] == "REPLACED" else 0.0)
    if len(xs) >= 3:
        mx = sum(xs) / len(xs)
        my = sum(ys) / len(ys)
        cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        vx = sum((x - mx) ** 2 for x in xs) ** 0.5
        vy = sum((y - my) ** 2 for y in ys) ** 0.5
        r = cov / (vx * vy) if vx > 0 and vy > 0 else float("nan")
        ax.set_title(f"Observed style shift predicts the verdict (pooled r = {r:.2f})", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_distance(numbers: dict, path: Path) -> None:
    distances = json.loads((REPORT / "measured_distances.json").read_text())
    fig, ax = plt.subplots(figsize=(5.0, 3.8))
    markers = {"diners": "o", "publicgoods": "s"}
    for game in GAMES:
        if game not in numbers:
            continue
        xs, ys, es = [], [], []
        for arm in ARMS:
            a = numbers[game]["arms"].get(arm)
            if not a:
                continue
            xs.append(distances[arm]["combined"])
            ys.append(a["p_replaced"])
            es.append(
                [
                    max(0.0, a["p_replaced"] - a["wilson_lo"]),
                    max(0.0, a["wilson_hi"] - a["p_replaced"]),
                ]
            )
        ax.errorbar(xs, ys, yerr=list(zip(*es)), marker=markers[game], color="#111827",
                    capsize=4, linewidth=1.2, label=GAME_LABELS[game])
        for x, y, arm in zip(xs, ys, [a for a in ARMS if numbers[game]["arms"].get(a)]):
            ax.annotate(arm, (x, y), textcoords="offset points", xytext=(6, -10), fontsize=8)
    ax.set_xlabel("Measured style distance from the base model (combined, bits)")
    ax.set_ylabel("P(REPLACED)")
    ax.set_ylim(-0.05, 1.1)
    ax.set_xlim(-0.03, 0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    numbers = load_numbers()
    fig_detection(numbers, FIGS / "fig1_detection.png")
    fig_style_shift(numbers, FIGS / "fig2_style_shift.png")
    fig_distance(numbers, FIGS / "fig3_distance.png")
    print("wrote", ", ".join(str(p.name) for p in sorted(FIGS.glob("*.png"))))


if __name__ == "__main__":
    main()
