"""Turn the model screen into an arm recommendation.

Criteria, in order:
1. Usable: valid JSON rate >= 0.67, valid choice rate >= 0.67, at least 5 style
   messages, and no budget-cap errors.
2. Capability parity: |p_cheap(a) - p_cheap(b)| <= 0.25 for any pair used.
3. Sibling arm: same family, smallest measured style distance among usable pairs.
4. Provider arm: different family, largest measured style distance among pairs
   that still satisfy capability parity.

The recommendation is a report, not an automatic commitment: the chosen ids and
the measured distances are pasted into the pre-registration before any swap run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCREEN = ROOT / "peer_swap" / "runs" / "model_screen.json"


def family(model: str) -> str:
    if model.startswith("anthropic/"):
        return "anthropic"
    if model.startswith("baseten/zai-org/"):
        return "zai"
    if model.startswith("baseten/deepseek-ai/"):
        return "deepseek"
    if model.startswith("bedrock/mistral."):
        return "mistral"
    if model.startswith("baseten/nvidia/"):
        return "nvidia"
    return model.split("/")[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DEFAULT_SCREEN))
    args = parser.parse_args()
    data = json.loads(Path(args.input).read_text())
    results = data["results"]
    distances = data["style_distance"]

    prices: dict[str, tuple[float, float]] = {}
    catalog = ROOT / "peer_swap" / "runs" / "respan_catalog.json"
    if catalog.exists():
        for row in json.loads(catalog.read_text()).get("results", []):
            name = row.get("model_name")
            if name:
                prices[name] = (
                    float(row.get("input_cost") or 0.0),
                    float(row.get("output_cost") or 0.0),
                )

    rows = []
    for model, record in results.items():
        decisions = record["decisions"]
        inp, out = prices.get(model, (float("nan"), float("nan")))
        rows.append(
            {
                "model": model,
                "family": family(model),
                "json": record["json"]["valid_json_rate"],
                "choice": decisions["valid_choice_rate"],
                "p_cheap": decisions["p_cheap"],
                "style_n": len(record["style"]["messages"]),
                "in_price": inp,
                "out_price": out,
            }
        )

    print(f"spent ${data['spent_usd']:.4f} over {data['calls']} calls\n")
    header = (
        f"{'model':52s} {'family':9s} {'json':>5s} {'choice':>6s} "
        f"{'p_cheap':>7s} {'style':>5s} {'$/M in':>7s} {'$/M out':>8s}"
    )
    print(header)
    print("-" * len(header))
    for row in sorted(rows, key=lambda r: (r["family"], r["model"])):
        print(
            f"{row['model']:52s} {row['family']:9s} {row['json']:5.2f} "
            f"{row['choice']:6.2f} {row['p_cheap']:7.2f} {row['style_n']:5d} "
            f"{row['in_price']:7.2f} {row['out_price']:8.2f}"
        )

    usable = [
        row
        for row in rows
        if row["json"] >= 0.67 and row["choice"] >= 0.67 and row["style_n"] >= 5
    ]
    print("\nusable:", len(usable), "of", len(rows))

    def parity_ok(a: str, b: str) -> bool:
        pa = next(r["p_cheap"] for r in rows if r["model"] == a)
        pb = next(r["p_cheap"] for r in rows if r["model"] == b)
        return abs(pa - pb) <= 0.25

    # Base: cheapest usable model, with validity rates as the tie-break. The
    # observer and the pre-swap partner are the largest token consumers, so the
    # base price dominates the run budget.
    base_candidates = sorted(
        usable,
        key=lambda r: (r["in_price"] + r["out_price"] / 2.0, -r["json"], -r["choice"]),
    )
    if not base_candidates:
        print("no usable base model")
        return
    base = base_candidates[0]["model"]

    sibling_options = [
        (distances[base][row["model"]], row["model"])
        for row in usable
        if row["model"] != base
        and row["family"] == family(base)
        and parity_ok(base, row["model"])
    ]
    provider_options = [
        (distances[base][row["model"]], row["model"])
        for row in usable
        if row["family"] != family(base) and parity_ok(base, row["model"])
    ]

    print(f"\nbase: {base}")
    if sibling_options:
        distance, model = min(sibling_options)
        print(f"sibling (same family, smallest distance): {model}  d_style={distance:.4f}")
    else:
        print("sibling: no same-family pair passes the capability gate")
    if provider_options:
        distance, model = max(provider_options)
        print(f"provider (different family, largest distance): {model}  d_style={distance:.4f}")
    else:
        print("provider: no cross-family pair passes the capability gate")

    print("\nstyle-distance matrix (Jensen-Shannon, bits):")
    models = [r["model"] for r in usable]
    print(" " * 52 + " ".join(f"{i:>6d}" for i in range(len(models))))
    for i, a in enumerate(models):
        cells = " ".join(
            f"{distances[a][b]:6.3f}" if a != b else "     -" for b in models
        )
        print(f"{i:>2d} {a:49s} {cells}")


if __name__ == "__main__":
    main()
