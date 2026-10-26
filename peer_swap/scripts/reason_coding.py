"""Coarse, transparent coding of the observer's stated reasons.

Each verdict reason is tagged for mentions of style, behaviour, arithmetic or
other cues. Multi-label: a reason can cite both style and behaviour. The coding
is keyword-based and saved verbatim so a reader can audit it; it is descriptive,
not a validated annotation scheme.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "peer_swap" / "runs"
REPORT = ROOT / "peer_swap" / "report"
GAMES = ("diners", "publicgoods")

PATTERNS = {
    "style": re.compile(
        r"\b(style|tone|phras\w*|wording|word choice|vocabulary|format\w*|verbos\w*|"
        r"writing|language|expression|sentence|capitali[sz]\w*)\b",
        re.I,
    ),
    "behaviour": re.compile(
        r"\b(behavio\w*|announc\w*|action\w*|choice|chose|choos\w*|play\w*|"
        r"strateg\w*|consistent|inconsistent|promise\w*|contribut\w*|cooperat\w*|"
        r"defect\w*|decision\w*|pattern\w*)\b",
        re.I,
    ),
    "arithmetic": re.compile(
        r"\b(arith\w*|payoff\w*|calculation|calculate\w*|math\w*|numeric\w*|"
        r"figure\w*|numbers?|mistake\w*|error\w*|miscalculat\w*)\b|\$",
        re.I,
    ),
}


def main() -> None:
    coded = []
    for game in GAMES:
        for arm in ("control", "sibling", "provider"):
            path = RUNS / f"verdicts_{game}_{arm}.json"
            if not path.exists():
                path = RUNS / f"verdicts_{arm}.json"
            if not path.exists():
                continue
            for row in json.loads(path.read_text()):
                reason = row.get("reason") or ""
                tags = [name for name, pattern in PATTERNS.items() if pattern.search(reason)]
                correct = (
                    row["verdict"] == ("SAME" if arm == "control" else "REPLACED")
                ) if row.get("verdict") else False
                coded.append(
                    {
                        "game": game,
                        "arm": arm,
                        "trial_id": row["trial_id"],
                        "verdict": row.get("verdict"),
                        "correct": correct,
                        "tags": tags or ["other"],
                        "reason": reason,
                    }
                )

    def counts(rows: list[dict]) -> dict[str, int]:
        counter: Counter = Counter()
        for row in rows:
            for tag in row["tags"]:
                counter[tag] += 1
        return dict(counter)

    summary = {
        "n": len(coded),
        "overall": counts(coded),
        "correct": counts([r for r in coded if r["correct"]]),
        "incorrect": counts([r for r in coded if not r["correct"]]),
        "by_arm": {arm: counts([r for r in coded if r["arm"] == arm]) for arm in ("control", "sibling", "provider")},
        "by_game": {game: counts([r for r in coded if r["game"] == game]) for game in GAMES},
        "coded": coded,
    }
    out = REPORT / "reason_coding.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"coded {len(coded)} reasons")
    print("overall:", summary["overall"])
    print("correct:", summary["correct"])
    print("incorrect:", summary["incorrect"])
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
