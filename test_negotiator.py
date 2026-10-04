"""Evaluate a self-play Transformer checkpoint against scripted opponents."""

from __future__ import annotations

import argparse
from itertools import combinations_with_replacement
from pathlib import Path
import random
import shutil

import numpy as np
import torch

from envs.env import AOPEnv
from policy import Transformer_Policy
from results import result_path, row_from_info, write_results
from selfplay.checkpoint import load_checkpoint
from selfplay.entry import PoolEntry
from train import CASES, SCRIPTED, normalize_case


def _device(value):
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    if str(value).startswith("cuda") and not torch.cuda.is_available():
        value = "cpu"
    return torch.device(value)


def evaluate(model_path, agents, issues, episodes=100, seed=0, device="auto",
             noise=False, deterministic=False, case=None, export_root=None):
    checkpoint = load_checkpoint(model_path, "cpu")
    config = checkpoint["config"]
    model_dir = Path(model_path)
    if model_dir.is_file():
        model_dir = model_dir.parent
    requested_case = normalize_case(case or config["utility_cases"][0])
    if requested_case not in CASES:
        raise ValueError(f"Unknown utility case: {requested_case}")
    if config["model_type"] == "expert" and any(issue not in config["issues"] for issue in issues):
        raise ValueError("Expert checkpoint can only evaluate its training domain")
    run_device = _device(device)
    first_env = AOPEnv(issues[0], test=True, scale=config["scale"], case=requested_case, add_noise=noise)
    model = Transformer_Policy(
        first_env.observation_space,
        first_env.action_space,
        run_device,
        domain=first_env.domain,
        issues=first_env.issues,
        is_acceptable=True,
        decoder_only=config["decoder_only"],
        scale=config["scale"],
        decoder_num=config["decoder_num"],
    ).to(run_device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    first_env.close()
    pairs = [tuple(agents)] if len(agents) == 2 else list(combinations_with_replacement(agents, 2))
    result_files = []
    for issue in issues:
        for pair in pairs:
            opponents = tuple(PoolEntry(f"scripted-{name}", "scripted", name) for name in pair)
            rows = []
            for episode in range(episodes):
                episode_seed = seed + episode
                random.seed(episode_seed)
                np.random.seed(episode_seed)
                torch.manual_seed(episode_seed)
                env = AOPEnv(issue, test=True, scale=config["scale"], case=requested_case, add_noise=noise)
                model.set_domain(issue, env.issues, env.action_space)
                observation, _ = env.reset(
                    seed=episode_seed, options={"opponent": pair, "case": requested_case}
                )
                done = False
                while not done:
                    with torch.no_grad():
                        tensor = torch.as_tensor(observation[None], device=run_device, dtype=torch.float32)
                        action, _ = model.mode(tensor) if deterministic else model.sample(tensor)[:2]
                    observation, _, terminated, truncated, info = env.step(int(action.item()))
                    done = terminated or truncated
                rows.append(row_from_info(
                    info, opponents, episode_seed, config["score_length_weight"]
                ))
                env.close()
            path = result_path(model_dir, pair, issue, deterministic, noise)
            write_results(path, rows)
            result_files.append(path)
            if export_root:
                target = (
                    Path(export_root) / config["model_type"] / f"{pair[0]}-{pair[1]}"
                    / issue / requested_case / path.name
                )
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
    return result_files


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", "--model", "-m", required=True)
    parser.add_argument("--agents", "-a", nargs="+", required=True, choices=SCRIPTED)
    parser.add_argument("--issues", "--issue", "-i", nargs="+", required=True)
    parser.add_argument("--episodes", "-e", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--noise", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--case", choices=CASES + ("1", "2", "3"))
    parser.add_argument("--export-root")
    args = parser.parse_args(argv)
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    for path in evaluate(
        args.model_path, args.agents, args.issues, args.episodes, seed=args.seed,
        device=args.device, noise=args.noise, deterministic=args.deterministic,
        case=args.case, export_root=args.export_root,
    ):
        print(path)


if __name__ == "__main__":
    main()
