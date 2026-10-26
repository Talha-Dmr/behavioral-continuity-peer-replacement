"""Compute actual API spend from the harness usage lines and the probe outputs."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOGS = [ROOT / "peer_swap" / "runs_main.log", ROOT / "peer_swap" / "runs_publicgoods_v2.log", ROOT / "peer_swap" / "runs_observer_glm.log"]
PRICES = {
    "anthropic/claude-haiku-5-5": (0.10, 0.50),
    "anthropic/claude-haiku-4-5": (1.00, 5.00),
    "baseten/nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B": (0.20, 0.50),
    "baseten/zai-org/GLM-5.3-Flash": (0.15, 0.50),
    "openai/gpt-5-mini": (0.25, 2.00),  # assumed: not in catalog page 1
}
OUT = ROOT / "peer_swap" / "report" / "cost_summary.json"


def main() -> None:
    usage = defaultdict(lambda: {"requests": 0, "prompt": 0, "completion": 0})
    for log in LOGS:
        if not log.exists():
            continue
        for line in log.read_text(errors="ignore").splitlines():
            if "USAGE_JSON=" not in line:
                continue
            payload = json.loads(line.split("USAGE_JSON=", 1)[1])
            for model, values in payload.items():
                bucket = usage[model]
                bucket["requests"] += int(values.get("requests", 0))
                bucket["prompt"] += int(values.get("prompt_tokens", 0))
                bucket["completion"] += int(values.get("completion_tokens", 0))

    per_model = {}
    total = 0.0
    for model, values in usage.items():
        pin, pout = PRICES.get(model, (0.0, 0.0))
        cost = values["prompt"] * pin / 1e6 + values["completion"] * pout / 1e6
        total += cost
        per_model[model] = {**values, "cost_usd": round(cost, 4)}

    probe = 0.0
    for name in ("active_probe_factual.json", "active_probe.json"):
        path = ROOT / "peer_swap" / "runs" / name
        if path.exists():
            probe += float(json.loads(path.read_text()).get("spent_usd", 0.0))

    selection = 0.2060 + 0.1630  # final screen + refinement
    result = {
        "swap_runs_usd": round(total, 4),
        "swap_calls": sum(v["requests"] for v in per_model.values()),
        "per_model": per_model,
        "selection_usd": selection,
        "active_probe_usd": round(probe, 4),
        "project_total_usd": round(total + selection + probe, 4),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
