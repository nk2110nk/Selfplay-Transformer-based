"""Shared seven-column negotiation result contract plus self-play metadata."""

from __future__ import annotations

import csv
from pathlib import Path

from selfplay.pfsp import negotiation_score

FIELDS = (
    "my_util", "opp_util1", "opp_util2", "social", "nash", "agreement", "step",
    "style", "opponent1_id", "opponent2_id", "opponent1_source", "opponent2_source",
    "negotiation_score", "seed", "domain", "case",
)


def result_path(model_dir, agents, issue, deterministic=False, noise=False):
    letter = lambda value: "T" if value else "F"
    return (
        Path(model_dir) / "csv" / f"{agents[0]}-{agents[1]}" / issue
        / f"det={deterministic}_noise={noise}"
        / f"{issue}-{agents[0]}-{agents[1]}-d{letter(deterministic)}-n{letter(noise)}.tsv"
    )


def row_from_info(info, opponents, seed, length_weight=-0.005, style="neutral"):
    state = info["state"]
    my_util = float(info["agent_utility"])
    opp1 = float(info["opponent1_utility"])
    opp2 = float(info["opponent2_utility"])
    return {
        "my_util": my_util,
        "opp_util1": opp1,
        "opp_util2": opp2,
        "social": my_util + opp1 + opp2,
        "nash": my_util * opp1 * opp2,
        "agreement": state["agreement"],
        "step": info["length"],
        "style": style,
        "opponent1_id": opponents[0].id,
        "opponent2_id": opponents[1].id,
        "opponent1_source": opponents[0].kind,
        "opponent2_source": opponents[1].kind,
        "negotiation_score": negotiation_score(
            info["agreement"], my_util, info["length"], length_weight
        ),
        "seed": seed,
        "domain": info["domain"],
        "case": info["case"],
    }


def write_results(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)
