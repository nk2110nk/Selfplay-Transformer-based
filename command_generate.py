"""Generate evaluation commands from managed self-play model directories."""

from __future__ import annotations

import argparse
import json
from itertools import combinations_with_replacement
from pathlib import Path
import shlex


def commands(results_root, episodes, export_root=None):
    generated = []
    for checkpoint in sorted(Path(results_root).glob("**/TransformerSelfPlay_Negotiator/checkpoint.pt")):
        model_dir = checkpoint.parent
        config_path = model_dir / "config.json"
        if not config_path.is_file():
            continue
        config = json.loads(config_path.read_text(encoding="utf-8"))
        agents = config["agents"]
        pairs = [tuple(agents)] if len(agents) == 2 and config["model_type"] == "expert" else list(combinations_with_replacement(agents, 2))
        for pair in pairs:
            for case in config["utility_cases"]:
                parts = [
                    "python", "test_negotiator.py", "-m", str(model_dir),
                    "-a", *pair, "-i", *config["issues"], "--episodes", str(episodes),
                    "--case", case,
                ]
                if export_root:
                    parts.extend(["--export-root", export_root])
                generated.append(" ".join(shlex.quote(value) for value in parts))
    return generated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_root")
    parser.add_argument("--output", "-o", default="run_command/run_test.sh")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--export-root")
    args = parser.parse_args()
    lines = commands(args.results_root, args.episodes, args.export_root)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text("#!/usr/bin/env bash\nset -euo pipefail\n\n" + "\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(f"generated:{len(lines)}:{output}")


if __name__ == "__main__":
    main()
