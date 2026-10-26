"""Analyse the model screen: separability, capability parity, arm choice.

Two additions over the raw screen:

- **Separability.** A leave-one-out nearest-centroid classifier over the style
  messages asks whether models can be told apart from their text at all. If it
  is near chance, the swap experiment cannot work through style, whatever the
  verdict rates say.
- **Provider pairs.** The same base model served by two different providers is
  the cleanest "different provider" arm, because the model is held constant.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "peer_swap" / "runs" / "model_screen_v3.json"
CATALOG = ROOT / "peer_swap" / "runs" / "respan_catalog.json"

STOP = set("a an the and or but if then so of to in on at for with from by is are was were be been it its this that these those i you we they he she my our your their".split())


def tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z']+", text.lower()) if t not in STOP]


def tfidf_vectors(messages: list[tuple[str, str]]) -> dict[str, list[float]]:
    """Binary term vectors weighted by inverse document frequency."""
    docs = [(model, Counter(tokens(text))) for model, text in messages]
    df: Counter = Counter()
    for _, counts in docs:
        df.update(counts.keys())
    n = len(docs)
    vocab = sorted(df)
    index = {token: i for i, token in enumerate(vocab)}
    vectors: dict[str, list[float]] = {}
    for i, (model, counts) in enumerate(docs):
        vec = [0.0] * len(vocab)
        for token, count in counts.items():
            vec[index[token]] = (1.0 + math.log(count)) * math.log((1 + n) / (1 + df[token]))
        vectors[f"{model}::{i}"] = vec
    return vectors


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


def separability(messages_by_model: dict[str, list[str]]) -> dict[str, float]:
    """Leave-one-out nearest-centroid accuracy over models with >= 5 messages."""
    usable = {m: msgs for m, msgs in messages_by_model.items() if len(msgs) >= 5}
    if len(usable) < 2:
        return {"models": float(len(usable)), "accuracy": float("nan"), "chance": float("nan")}
    flat = [(model, text) for model, msgs in usable.items() for text in msgs]
    vectors = tfidf_vectors(flat)
    correct = 0
    for i, (model, _) in enumerate(flat):
        # Build centroids excluding the held-out document.
        grouped: dict[str, list[list[float]]] = defaultdict(list)
        for j, (other_model, _) in enumerate(flat):
            if i == j:
                continue
            grouped[other_model].append(vectors[f"{other_model}::{j}"])
        best_model, best_score = None, -2.0
        held = vectors[f"{model}::{i}"]
        for other_model, vecs in grouped.items():
            centroid = [sum(col) / len(vecs) for col in zip(*vecs)]
            score = cosine(held, centroid)
            if score > best_score:
                best_model, best_score = other_model, score
        correct += int(best_model == model)
    return {
        "models": float(len(usable)),
        "accuracy": correct / len(flat),
        "chance": 1.0 / len(usable),
    }


def base_model(model: str) -> str:
    """Normalise a provider-prefixed id to the underlying model name."""
    name = model
    for prefix in ("baseten/", "bedrock/"):
        if name.startswith(prefix):
            name = name[len(prefix) :]
    if name.startswith("openai/"):
        name = name[len("openai/") :]
    name = name.replace("openai.", "")
    name = re.sub(r"-1:0$", "", name)
    name = name.replace(".", "/")
    return name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    args = parser.parse_args()

    data = json.loads(Path(args.input).read_text())
    results = data["results"]
    distances = data["style_distance"]

    prices: dict[str, tuple[float, float]] = {}
    for row in json.loads(CATALOG.read_text()).get("results", []):
        if row.get("model_name"):
            prices[row["model_name"]] = (
                float(row.get("input_cost") or 0.0),
                float(row.get("output_cost") or 0.0),
            )

    rows = []
    for model, record in results.items():
        inp, out = prices.get(model, (float("nan"), float("nan")))
        rows.append(
            {
                "model": model,
                "json": record["json"]["valid_json_rate"],
                "choice": record["decisions"]["valid_choice_rate"],
                "p_cheap": record["decisions"]["p_cheap"],
                "style_n": len(record["style"]["messages"]),
                "in_price": inp,
                "out_price": out,
            }
        )

    usable = [
        row
        for row in rows
        if row["json"] >= 0.67 and row["choice"] >= 0.67 and row["style_n"] >= 5
    ]

    print(f"spent ${data['spent_usd']:.4f} over {data['calls']} calls")
    print(f"usable: {len(usable)} / {len(rows)}\n")
    header = f"{'model':55s} {'json':>5s} {'choice':>6s} {'p_cheap':>7s} {'n':>3s} {'$/M in':>7s} {'$/M out':>8s}"
    print(header)
    print("-" * len(header))
    for row in sorted(usable, key=lambda r: (r["in_price"], r["model"])):
        print(
            f"{row['model']:55s} {row['json']:5.2f} {row['choice']:6.2f} "
            f"{row['p_cheap']:7.2f} {row['style_n']:3d} {row['in_price']:7.2f} {row['out_price']:8.2f}"
        )

    messages_by_model = {
        model: results[model]["style"]["messages"] for model in results
    }
    sep = separability(messages_by_model)
    print(
        f"\nstyle separability: accuracy={sep['accuracy']:.3f} "
        f"(chance={sep['chance']:.3f}, {int(sep['models'])} models)"
    )

    # Same base model on different providers.
    provider_pairs = []
    by_base: dict[str, list[str]] = defaultdict(list)
    for row in usable:
        by_base[base_model(row["model"])].append(row["model"])
    for base, models in by_base.items():
        if len(models) > 1:
            a, b = models[:2]
            provider_pairs.append((a, b, distances[a][b]))
    print("\nsame base model, different provider:")
    for a, b, d in provider_pairs:
        pa = next(r["p_cheap"] for r in usable if r["model"] == a)
        pb = next(r["p_cheap"] for r in usable if r["model"] == b)
        print(f"  {a}  vs  {b}  d_style={d:.4f}  |dP(CHEAP)|={abs(pa - pb):.2f}")

    def parity(a: str, b: str) -> float:
        pa = next(r["p_cheap"] for r in usable if r["model"] == a)
        pb = next(r["p_cheap"] for r in usable if r["model"] == b)
        return abs(pa - pb)

    base_row = min(usable, key=lambda r: r["in_price"] + r["out_price"] / 2.0)
    base = base_row["model"]

    def fam(model: str) -> str:
        if model.startswith("anthropic/"):
            return "anthropic"
        if "mistral" in model:
            return "mistral"
        if "zai-org" in model:
            return "zai"
        if "deepseek" in model:
            return "deepseek"
        if "nvidia" in model.lower() or "nemotron" in model.lower():
            return "nvidia"
        if "gpt-oss" in model:
            return "gpt-oss"
        return model.split("/")[0]

    siblings = [
        (distances[base][row["model"]], row["model"])
        for row in usable
        if row["model"] != base and fam(row["model"]) == fam(base) and parity(base, row["model"]) <= 0.25
    ]
    providers = [
        (distances[base][row["model"]], row["model"])
        for row in usable
        if fam(row["model"]) != fam(base) and parity(base, row["model"]) <= 0.25
    ]

    print(f"\ncheapest usable base: {base}")
    print(
        "sibling: " + (f"{min(siblings)[1]} d={min(siblings)[0]:.4f}" if siblings else "none passes")
    )
    print(
        "provider (different family): "
        + (f"{max(providers)[1]} d={max(providers)[0]:.4f}" if providers else "none passes")
    )


if __name__ == "__main__":
    main()
