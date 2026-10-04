"""Train a Transformer PPO negotiator with persistent PFSP self-play."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
from pathlib import Path

from selfplay.checkpoint import load_checkpoint
from selfplay.trainer import SelfPlayPPOTrainer

SCRIPTED = (
    "Boulware", "Linear", "Conceder", "TitForTat1", "TitForTat2",
    "AgentK", "HardHeaded", "Atlas3", "AgentGG",
)
CASES = ("case1", "case2", "case3")


def normalize_case(value):
    value = str(value)
    return value if value.startswith("case") else f"case{value}"


def build_model_dir(save_path, issues, agents, resume=None):
    if resume:
        path = Path(resume)
        return path if path.is_dir() else path.parent
    root = Path(save_path)
    if root.name == "TransformerSelfPlay_Negotiator":
        return root
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return root / f"{'-'.join(issues)}_{'-'.join(agents)}" / f"{stamp}-TA" / "TransformerSelfPlay_Negotiator"


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--agents", "-a", nargs="+", choices=SCRIPTED)
    parser.add_argument("--issue", "--issues", "-i", nargs="+")
    parser.add_argument("--model-type", choices=("expert", "general"), default="expert")
    parser.add_argument("--save-path", "--save_path", "-sp", default="results")
    parser.add_argument("--resume", "--model", "-m")
    parser.add_argument("--total-timesteps", "--timesteps", "-t", type=int, default=300_000)
    parser.add_argument("--num-envs", "--n-envs", "-n", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--scale", "-s", default="small")
    parser.add_argument("--decoder-only", "--decoder_only", "-do", action="store_true")
    parser.add_argument("--decoder-num", "--decoder_num", "-dn", type=int, default=1)
    parser.add_argument("--learning-rate", "--learning_rate", "-lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", "--batch_size", "-bs", type=int, default=64)
    parser.add_argument("--horizon", "-hr", type=int, default=2048)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", "--clip_range", "-cr", type=float, default=0.2)
    parser.add_argument("--entropy-coef", "--entropy_coef", "-ec", type=float, default=0.0)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--no-normalize-advantage", action="store_true")
    parser.add_argument("--pool-path", default="pool/pool.json")
    parser.add_argument("--max-pool-size", type=int, default=16)
    parser.add_argument("--pfsp-alpha", type=float, default=2.0)
    parser.add_argument("--uniform-mix", type=float, default=0.1)
    parser.add_argument("--allow-duplicate-opponents", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--self-play-probability", type=float, default=0.0)
    parser.add_argument("--scripted-probability", type=float, default=0.5)
    parser.add_argument("--snapshot-probability", type=float, default=0.5)
    parser.add_argument("--snapshot-freq", type=int, default=10_000)
    parser.add_argument("--pool-eval-freq", type=int, default=10_000)
    parser.add_argument("--pool-eval-episodes", type=int, default=4)
    parser.add_argument("--dominance-tolerance", type=float, default=0.01)
    parser.add_argument("--min-agreement-rate", type=float, default=0.0)
    parser.add_argument("--score-length-weight", type=float, default=-0.005)
    parser.add_argument("--utility-cases", nargs="+", default=["case1"])
    parser.add_argument("--opponent-noise", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)

    if not args.resume and (not args.agents or not args.issue):
        parser.error("--agents and --issue are required for a new run")
    args.utility_cases = list(dict.fromkeys(normalize_case(value) for value in args.utility_cases))
    if any(value not in CASES for value in args.utility_cases):
        parser.error(f"--utility-cases must be selected from {', '.join(CASES)}")
    numeric_positive = (
        args.total_timesteps, args.num_envs, args.batch_size, args.horizon,
        args.n_epochs, args.pool_eval_freq, args.snapshot_freq, args.pool_eval_episodes,
    )
    if any(value < 1 for value in numeric_positive):
        parser.error("Timesteps, environment counts, PPO sizes and frequencies must be positive")
    source = (args.self_play_probability, args.scripted_probability, args.snapshot_probability)
    if any(not math.isfinite(value) or value < 0 for value in source) or sum(source) <= 0:
        parser.error("Opponent source probabilities must be finite, non-negative and not all zero")
    if not 0 <= args.uniform_mix <= 1 or args.pfsp_alpha < 0:
        parser.error("Invalid PFSP parameters")
    if args.agents and args.max_pool_size < len(set(args.agents)) + 1:
        parser.error("Pool needs room for scripted opponents and at least one snapshot")
    if args.agents and not args.allow_duplicate_opponents and len(set(args.agents)) < 2 and args.self_play_probability == 0:
        parser.error("At least two available entries are required when duplicates are disabled")
    if args.issue and args.model_type == "expert" and len(args.issue) != 1:
        parser.error("Expert mode requires exactly one issue; use --model-type general otherwise")
    return args


def _new_config(args):
    return {
        "issues": list(args.issue),
        "agents": list(dict.fromkeys(args.agents)),
        "model_type": args.model_type,
        "seed": args.seed,
        "device": args.device,
        "scale": args.scale,
        "decoder_only": args.decoder_only,
        "decoder_num": args.decoder_num,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "horizon": args.horizon,
        "num_envs": args.num_envs,
        "n_epochs": args.n_epochs,
        "gamma": args.gamma,
        "gae_lambda": args.gae_lambda,
        "clip_range": args.clip_range,
        "normalize_advantage": not args.no_normalize_advantage,
        "entropy_coef": args.entropy_coef,
        "vf_coef": args.vf_coef,
        "max_grad_norm": args.max_grad_norm,
        "pool_path": args.pool_path,
        "max_pool_size": args.max_pool_size,
        "pfsp_alpha": args.pfsp_alpha,
        "uniform_mix": args.uniform_mix,
        "allow_duplicate_opponents": args.allow_duplicate_opponents,
        "self_play_probability": args.self_play_probability,
        "scripted_probability": args.scripted_probability,
        "snapshot_probability": args.snapshot_probability,
        "snapshot_freq": args.snapshot_freq,
        "pool_eval_freq": args.pool_eval_freq,
        "pool_eval_episodes": args.pool_eval_episodes,
        "dominance_tolerance": args.dominance_tolerance,
        "min_agreement_rate": args.min_agreement_rate,
        "score_length_weight": args.score_length_weight,
        "utility_cases": args.utility_cases,
        "opponent_noise": args.opponent_noise,
    }


def train(args):
    checkpoint = load_checkpoint(args.resume, "cpu") if args.resume else None
    if checkpoint:
        config = dict(checkpoint["config"])
        for supplied, key in ((args.issue, "issues"), (args.agents, "agents")):
            if supplied is not None and list(supplied) != config[key]:
                raise ValueError(f"Resume configuration differs for {key}: {supplied} != {config[key]}")
        config["device"] = args.device
        model_dir = build_model_dir(args.save_path, config["issues"], config["agents"], args.resume)
    else:
        config = _new_config(args)
        model_dir = build_model_dir(args.save_path, config["issues"], config["agents"])
        model_dir.mkdir(parents=True, exist_ok=True)
        temporary = model_dir / "config.json.tmp"
        temporary.write_text(json.dumps(config, indent=2), encoding="utf-8")
        temporary.replace(model_dir / "config.json")
    trainer = SelfPlayPPOTrainer(config, model_dir, checkpoint)
    return trainer.train(args.total_timesteps)


def main(argv=None):
    args = parse_args(argv)
    model_dir = train(args)
    print(f"saved_model:{model_dir / 'checkpoint.pt'}")


if __name__ == "__main__":
    main()
