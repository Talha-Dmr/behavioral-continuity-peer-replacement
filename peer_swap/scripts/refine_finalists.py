"""Re-measure the finalists with larger samples before freezing the arms.

The first screen used 6 decisions and 8 style messages. That is enough to
reject broken models but too noisy to order them: the same-family pair came out
with a larger style distance than several cross-family pairs. Here each
finalist gets 20 decisions (capability parity with a Wilson interval) and 16
style messages (distance and separability).
"""

from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parents[1]
sys.path.insert(0, str(SCRIPTS))

from screen_models import (  # noqa: E402
    Budget,
    load_prices,
    probe_decisions,
    probe_style,
    read_key,
    surface_features,
)

FINALISTS = [
    "anthropic/claude-haiku-5-5",          # cheapest usable, likely base
    "anthropic/claude-haiku-4-5",          # official same-family sibling
    "baseten/zai-org/GLM-5.3-Flash",       # cross-family, parity candidate
    "baseten/nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B",
    "anthropic/claude-sonnet-5-5",         # same family, different tier
    "baseten/deepseek-ai/DeepSeek-V4.1-Flash",
]

OUT = ROOT / "peer_swap" / "runs" / "model_refine.json"


def main() -> None:
    key = read_key()
    budget = Budget(0.40, load_prices())
    results: dict[str, dict] = {}

    def run(model: str) -> tuple[str, dict]:
        record = {
            "model": model,
            "decisions": probe_decisions(key, model, budget, n=20),
            "style": probe_style(key, model, budget, n=16),
        }
        record["surface"] = [surface_features(m) for m in record["style"]["messages"]]
        d = record["decisions"]
        print(
            f"{model}: p_cheap={d['p_cheap']:.2f} valid={d['valid_choice_rate']:.2f} "
            f"style_n={len(record['style']['messages'])}",
            flush=True,
        )
        return model, record

    with ThreadPoolExecutor(max_workers=3) as pool:
        for model, record in pool.map(run, FINALISTS):
            results[model] = record

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {"spent_usd": round(budget.spent, 6), "calls": budget.calls, "results": results},
            indent=2,
        )
    )
    print(f"\nspent ${budget.spent:.4f} over {budget.calls} calls")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
