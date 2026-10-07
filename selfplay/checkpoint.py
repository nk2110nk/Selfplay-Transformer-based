"""Versioned, atomic training checkpoint utilities."""

from __future__ import annotations

from pathlib import Path
import random

import numpy as np
import torch

CHECKPOINT_FORMAT_VERSION = 3


def random_states(rng):
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "sampler": rng.bit_generator.state,
    }


def restore_random_states(states, rng):
    random.setstate(states["python"])
    np.random.set_state(states["numpy"])
    torch.set_rng_state(states["torch"])
    if torch.cuda.is_available() and states.get("cuda") is not None:
        torch.cuda.set_rng_state_all(states["cuda"])
    rng.bit_generator.state = states["sampler"]


def save_checkpoint(model_dir, model, optimizer, pool, config, global_step,
                    episode, update_step, rng, schedule_state):
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    pool.save()
    state = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "architecture_version": model.ARCHITECTURE_VERSION,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "model_config": model.model_config(),
        "action_mapping": config["action_mapping"],
        "utility_cases": config["utility_cases"],
        "utility_orders": config["utility_orders"],
        "config": config,
        "global_step": int(global_step),
        "cumulative_timesteps": int(global_step),
        "episode": int(episode),
        "update_step": int(update_step),
        "pool_metadata": pool.to_dict(),
        "pool_path": str(pool.path.relative_to(model_dir)),
        "schedule_state": schedule_state,
        "random_states": random_states(rng),
    }
    destination = model_dir / "checkpoint.pt"
    temporary = model_dir / "checkpoint.pt.tmp"
    torch.save(state, temporary)
    temporary.replace(destination)
    return destination


def load_checkpoint(path, device="cpu"):
    path = Path(path)
    if path.is_dir():
        path = path / "checkpoint.pt"
    state = torch.load(path, map_location=device, weights_only=False)
    if state.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(f"Unsupported checkpoint format: {state.get('format_version')}")
    if state.get("architecture_version") != 2:
        raise ValueError(f"Unsupported Transformer architecture: {state.get('architecture_version')}")
    config = state.get("config", {})
    if state.get("utility_cases") != config.get("utility_cases"):
        raise ValueError("Checkpoint utility cases do not match its configuration")
    if state.get("utility_orders") != config.get("utility_orders"):
        raise ValueError("Checkpoint utility orders do not match its configuration")
    return state
