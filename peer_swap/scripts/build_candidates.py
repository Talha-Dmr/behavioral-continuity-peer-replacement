"""Build the model candidate list from the Respan catalog by a fixed rule.

Rule (declared before looking at screen results):
1. Managed by Respan, not BYOK, status active.
2. Text-chat models only (drop media generators by name).
3. Price ceiling: input <= $2/M and output <= $10/M. Above the ceiling the
   experiment's hundreds of calls become unbounded, so those models are
   excluded for budget, not for scientific reasons.
4. Collapse aliases of the same model: drop dated ids (a shorter id exists)
   and "-Fast" serving variants.

Writes peer_swap/runs/candidates.json with the kept list and every exclusion.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CATALOG = ROOT / "peer_swap" / "runs" / "respan_catalog.json"
OUT = ROOT / "peer_swap" / "runs" / "candidates.json"

NON_CHAT = ("veo", "voxtral", "embedding", "rerank", "image", "tts", "whisper")
MAX_INPUT = 2.0
MAX_OUTPUT = 10.0


def price(row: dict, key: str) -> float:
    return float(row.get(key) or 0.0)


def canonical_key(name: str) -> str:
    """Strip a dated suffix and a Fast serving suffix to find aliases."""
    key = re.sub(r"-Fast$", "", name)
    key = re.sub(r"-\d{4,8}$", "", key)
    return key


def main() -> None:
    rows = json.loads(CATALOG.read_text()).get("results", [])
    kept: list[dict] = []
    excluded: list[dict] = []

    managed = [
        row
        for row in rows
        if row.get("is_managed") and not row.get("is_byok") and row.get("status") == "active"
    ]
    for row in rows:
        if row not in managed:
            excluded.append(
                {
                    "model": row.get("model_name"),
                    "reason": "not managed/active on Respan credits (BYOK, deprecated, or external)",
                }
            )

    text = []
    for row in managed:
        name = row.get("model_name", "")
        if any(token in name.lower() for token in NON_CHAT):
            excluded.append({"model": name, "reason": "non-chat media model"})
        else:
            text.append(row)

    affordable = []
    for row in text:
        if price(row, "input_cost") <= MAX_INPUT and price(row, "output_cost") <= MAX_OUTPUT:
            affordable.append(row)
        else:
            excluded.append(
                {
                    "model": row.get("model_name"),
                    "reason": f"price above ceiling (in {row.get('input_cost')}, out {row.get('output_cost')})",
                }
            )

    # Keep one id per (canonical key), preferring the shortest name (undated,
    # non-Fast). Ties are broken by the cheaper price.
    groups: dict[str, list[dict]] = {}
    for row in affordable:
        groups.setdefault(canonical_key(row["model_name"]), []).append(row)

    for key, members in groups.items():
        members.sort(key=lambda row: (len(row["model_name"]), price(row, "input_cost")))
        kept.append(members[0])
        for drop in members[1:]:
            excluded.append(
                {"model": drop.get("model_name"), "reason": f"alias of {members[0]['model_name']}"}
            )

    kept.sort(key=lambda row: (price(row, "input_cost"), price(row, "output_cost"), row["model_name"]))
    output = {
        "rule": {
            "managed_only": True,
            "exclude_byok": True,
            "active_only": True,
            "non_chat_tokens": list(NON_CHAT),
            "max_input_cost_per_m": MAX_INPUT,
            "max_output_cost_per_m": MAX_OUTPUT,
            "dedupe": "strip dated suffix and -Fast, keep shortest id",
        },
        "candidates": [row["model_name"] for row in kept],
        "excluded": excluded,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(output, indent=2))

    print(f"candidates: {len(kept)}")
    for row in kept:
        print(f"  {row['model_name']:55s} in {row.get('input_cost')} out {row.get('output_cost')}")
    print(f"excluded: {len(excluded)} (reasons in {OUT})")


if __name__ == "__main__":
    main()
