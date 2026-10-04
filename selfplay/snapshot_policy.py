"""Frozen, independently observed Transformer snapshot opponents."""

from __future__ import annotations

import copy
from pathlib import Path

import torch

from envs.observer import EmbeddedObserveHistory
from envs.rl_negotiator import PolicyNegotiator
from policy import Transformer_Policy

SNAPSHOT_FORMAT_VERSION = 2


def build_policy(source, env, state_dict=None, device="cpu"):
    if isinstance(source, Transformer_Policy):
        config = source.model_config()
        policy = copy.deepcopy(source).to(device)
        policy.device = torch.device(device)
    else:
        config = dict(source)
        if config.get("architecture_version") != Transformer_Policy.ARCHITECTURE_VERSION:
            raise ValueError("Incompatible Transformer snapshot architecture")
        policy = Transformer_Policy(
            obs_space=env.observation_space,
            action_space=env.action_space,
            device=device,
            domain=env.domain,
            issues=env.issues,
            is_acceptable=True,
            decoder_only=config["decoder_only"],
            scale=config["scale"],
            decoder_num=config["decoder_num"],
        )
        if state_dict is None:
            raise ValueError("Snapshot state_dict is required")
        policy.load_state_dict(state_dict)
    policy.set_domain(env.domain, env.issues, env.action_space)
    policy.eval()
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    return policy


class SnapshotPolicyFactory:
    def __init__(self, checkpoint, device="cpu", deterministic=False):
        self.checkpoint = Path(checkpoint)
        self.device = device
        self.deterministic = deterministic
        # Snapshot bytes are immutable. Cache the CPU state while still
        # constructing an independent model and observer for every slot.
        self.state = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
        if self.state.get("format_version") != SNAPSHOT_FORMAT_VERSION:
            raise ValueError(f"Unsupported snapshot format: {self.state.get('format_version')}")

    def __call__(self, env, slot):
        state = self.state
        compatible = state.get("compatible_domains", [])
        if compatible and env.domain not in compatible:
            raise ValueError(f"Snapshot is incompatible with domain {env.domain}")
        mapping = state.get("action_mapping", {}).get(env.domain)
        if mapping is not None and mapping["action_count"] != env.action_space.n:
            raise ValueError(f"Snapshot action mapping is incompatible with domain {env.domain}")
        policy = build_policy(state["model_config"], env, state["model"], self.device)
        observer = EmbeddedObserveHistory(
            env.domain, env.issues, n_turn=env.n_steps, scale=state["model_config"]["scale"]
        )
        return PolicyNegotiator(
            policy,
            observer,
            deterministic=self.deterministic,
            name=f"{state.get('name', 'TransformerSnapshot')}-{slot}",
        )


class CurrentPolicyFactory:
    def __init__(self, model, device="cpu", deterministic=False):
        # Freeze immediately so later learner updates cannot affect this episode.
        self.model = copy.deepcopy(model).to(device)
        self.device = device
        self.deterministic = deterministic
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def __call__(self, env, slot):
        policy = build_policy(self.model, env, device=self.device)
        observer = EmbeddedObserveHistory(env.domain, env.issues, env.n_steps, scale=policy.scale)
        return PolicyNegotiator(
            policy, observer, deterministic=self.deterministic, name=f"CurrentTransformer-{slot}"
        )


def save_snapshot(model_dir, model, config, step, metrics):
    directory = Path(model_dir) / "pool" / "snapshots"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"snapshot-{int(step)}.pt"
    temporary = path.with_suffix(".pt.tmp")
    state = {
        "format_version": SNAPSHOT_FORMAT_VERSION,
        "architecture_version": model.ARCHITECTURE_VERSION,
        "name": f"TransformerSnapshot{int(step)}",
        "model": model.state_dict(),
        "model_config": model.model_config(),
        "issues": config["issues"],
        "model_type": config["model_type"],
        "case": config["utility_cases"],
        "compatible_domains": config["issues"],
        "action_mapping": config["action_mapping"],
        "benchmark": metrics,
        "global_step": int(step),
    }
    torch.save(state, temporary)
    temporary.replace(path)
    return path


def factory_for_entry(entry, model_dir, learner_model=None, device="cpu", deterministic=False):
    if entry.kind == "scripted":
        return entry.name
    if entry.kind == "snapshot":
        return SnapshotPolicyFactory(Path(model_dir) / entry.checkpoint_path, device, deterministic)
    if entry.kind == "self_play":
        if learner_model is None:
            raise ValueError("Current self-play requires the learner model")
        return CurrentPolicyFactory(learner_model, device, deterministic)
    raise ValueError(f"Unsupported opponent kind: {entry.kind}")
