"""Balanced pool evaluation with learning RNG isolation."""

from __future__ import annotations

import itertools
import random

import numpy as np
import torch

from envs.env import AOPEnv
from .pfsp import negotiation_score
from .snapshot_policy import factory_for_entry


def balanced_schedule(domains, anchors, cases, minimum_episodes):
    domains = list(dict.fromkeys(domains))
    anchors = list(dict.fromkeys(anchors))
    cases = list(dict.fromkeys(cases))
    if not domains or not anchors or not cases or minimum_episodes < 1:
        raise ValueError("Pool evaluation needs domains, anchors, cases and positive episodes")
    domain_cycles = max(1, (minimum_episodes + len(domains) - 1) // len(domains))
    contexts = itertools.cycle(itertools.product(anchors, cases, (0, 1)))
    schedule = []
    for _ in range(domain_cycles):
        for domain in domains:
            anchor, case, slot = next(contexts)
            schedule.append((domain, anchor, case, slot))
    return schedule


def run_episode(model, entry, anchor, slot, domain, case, model_dir, seed, device="cpu", target=None):
    env = AOPEnv(domain=domain, test=True, scale=model.scale, case=case, add_noise=False)
    model.set_domain(domain, env.issues, env.action_space)
    if target is None:
        target = factory_for_entry(entry, model_dir, learner_model=model, device=device, deterministic=True)
    opponents = [anchor, anchor]
    opponents[slot] = target
    observation, _ = env.reset(seed=seed, options={"opponent_factory": opponents, "case": case})
    terminated = False
    info = None
    while not terminated:
        with torch.no_grad():
            action, _ = model.mode(torch.as_tensor(observation[None], device=model.device, dtype=torch.float32))
        observation, _, terminated, truncated, info = env.step(int(action.item()))
        terminated = terminated or truncated
    env.close()
    return info


def _aggregate(rows, length_weight):
    if not rows:
        raise ValueError("Cannot aggregate an empty evaluation")
    by_domain = {}
    for row in rows:
        by_domain.setdefault(row["domain"], []).append(row)
    per_domain = []
    for domain_rows in by_domain.values():
        count = len(domain_rows)
        per_domain.append({
            "agreement_rate": sum(float(row["agreement"]) for row in domain_rows) / count,
            "agent_utility": sum(row["agent_utility"] for row in domain_rows) / count,
            "mean_length": sum(row["length"] for row in domain_rows) / count,
            "opponent1_utility": sum(row["opponent1_utility"] for row in domain_rows) / count,
            "opponent2_utility": sum(row["opponent2_utility"] for row in domain_rows) / count,
            "social_welfare": sum(row["social_welfare"] for row in domain_rows) / count,
        })
    count = len(per_domain)
    metrics = {
        key: sum(domain_metrics[key] for domain_metrics in per_domain) / count
        for key in per_domain[0]
    }
    metrics["negotiation_score"] = negotiation_score(
        metrics["agreement_rate"], metrics["agent_utility"], metrics["mean_length"], length_weight
    )
    return metrics


def evaluate_pool(model, pool, domains, cases, episodes, model_dir, seed,
                  length_weight=-0.005, step=0, device="cpu"):
    scripted = [entry.name for entry in pool.entries if entry.kind == "scripted"]
    if not scripted:
        raise ValueError("Pool needs a scripted anchor")
    states = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    was_training = model.training
    model.eval()
    measurements, all_rows = {}, []
    try:
        schedule = balanced_schedule(domains, scripted, cases, episodes)
        for entry_index, entry in enumerate(list(pool.entries)):
            rows = []
            target = factory_for_entry(
                entry, model_dir, learner_model=model, device=device, deterministic=True
            )
            for episode_index, (domain, anchor, case, slot) in enumerate(schedule):
                episode_seed = int(seed + step * 100003 + entry_index * 1009 + episode_index)
                info = run_episode(
                    model, entry, anchor, slot, domain, case, model_dir, episode_seed, device, target
                )
                row = dict(info)
                row.update({
                    "opponent_id": entry.id,
                    "anchor": anchor,
                    "slot": slot,
                    "evaluation_seed": episode_seed,
                })
                if slot == 0:
                    row.update({
                        "opponent1_id": entry.id,
                        "opponent2_id": f"scripted-{anchor}",
                        "opponent1_source": entry.kind,
                        "opponent2_source": "scripted",
                    })
                else:
                    row.update({
                        "opponent1_id": f"scripted-{anchor}",
                        "opponent2_id": entry.id,
                        "opponent1_source": "scripted",
                        "opponent2_source": entry.kind,
                    })
                rows.append(row)
                all_rows.append(row)
            metrics = _aggregate(rows, length_weight)
            measurements[entry.id] = metrics
            # Evaluation outcomes update only learner-observable pool fields.
            for row in rows:
                entry.record(row["agreement"], row["agent_utility"], row["length"])
        benchmark = _aggregate(all_rows, length_weight)
    finally:
        random.setstate(states["python"])
        np.random.set_state(states["numpy"])
        torch.set_rng_state(states["torch"])
        if torch.cuda.is_available() and states["cuda"] is not None:
            torch.cuda.set_rng_state_all(states["cuda"])
        if was_training:
            model.train()
    return measurements, benchmark, all_rows
