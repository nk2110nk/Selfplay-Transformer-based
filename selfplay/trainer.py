"""Synchronous PPO trainer backed by a persistent self-play opponent pool."""

from __future__ import annotations

from collections import defaultdict
import csv
import json
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn.functional as F

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # pragma: no cover
    SummaryWriter = None

from envs.env import AOPEnv
from policy import RolloutBuffer, Transformer_Policy
from results import write_results
from .checkpoint import restore_random_states, save_checkpoint
from .evaluator import evaluate_pool
from .pfsp import negotiation_score
from .pool import OpponentPool, is_dominant
from .snapshot_policy import factory_for_entry, save_snapshot


class SelfPlayPPOTrainer:
    def __init__(self, config, model_dir, checkpoint=None):
        self.config = dict(config)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.device = self._device(config["device"])
        self.rng = np.random.default_rng(config["seed"])
        random.seed(config["seed"])
        np.random.seed(config["seed"])
        torch.manual_seed(config["seed"])
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config["seed"])

        first_env = AOPEnv(
            config["issues"][0], scale=config["scale"], case=config["utility_cases"][0],
            add_noise=config["opponent_noise"],
        )
        self.model = Transformer_Policy(
            obs_space=first_env.observation_space,
            action_space=first_env.action_space,
            device=self.device,
            domain=first_env.domain,
            issues=first_env.issues,
            is_acceptable=True,
            decoder_only=config["decoder_only"],
            scale=config["scale"],
            decoder_num=config["decoder_num"],
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=config["learning_rate"], eps=1e-5
        )
        action_mapping = {
            first_env.domain: {
                "accept": 0,
                "bid_offset": 1,
                "bid_count": len(first_env.all_bids),
                "action_count": int(first_env.action_space.n),
                "observation_shape": list(first_env.observation_space.shape),
            }
        }
        first_env.close()
        for domain in config["issues"][1:]:
            metadata_env = AOPEnv(
                domain, scale=config["scale"], case=config["utility_cases"][0], add_noise=False
            )
            action_mapping[domain] = {
                "accept": 0,
                "bid_offset": 1,
                "bid_count": len(metadata_env.all_bids),
                "action_count": int(metadata_env.action_space.n),
                "observation_shape": list(metadata_env.observation_space.shape),
            }
            metadata_env.close()
        if "action_mapping" in self.config and self.config["action_mapping"] != action_mapping:
            raise ValueError("Checkpoint action mapping does not match the configured domains")
        self.config["action_mapping"] = action_mapping
        temporary_config = self.model_dir / "config.json.tmp"
        temporary_config.write_text(json.dumps(self.config, indent=2), encoding="utf-8")
        temporary_config.replace(self.model_dir / "config.json")

        pool_path = Path(config["pool_path"])
        if not pool_path.is_absolute():
            pool_path = self.model_dir / pool_path
        try:
            pool_path.relative_to(self.model_dir)
        except ValueError as error:
            raise ValueError("--pool-path must be inside the model directory") from error
        if checkpoint:
            self.pool = OpponentPool.load(pool_path, model_root=self.model_dir)
        else:
            self.pool = OpponentPool(
                root=pool_path.parent,
                scripted_names=config["agents"],
                max_size=config["max_pool_size"],
                pfsp_alpha=config["pfsp_alpha"],
                uniform_mix=config["uniform_mix"],
                allow_duplicates=config["allow_duplicate_opponents"],
                self_play_probability=config["self_play_probability"],
                scripted_probability=config["scripted_probability"],
                snapshot_probability=config["snapshot_probability"],
                path=pool_path,
                model_root=self.model_dir,
            )
            self.pool.save(pool_path)

        self.global_step = 0
        self.episode = 0
        self.update_step = 0
        self.schedule_index = 0
        self.next_pool_eval = config["pool_eval_freq"]
        self.next_snapshot = config["snapshot_freq"]
        self.source_selected = {"scripted": 0, "snapshot": 0, "self_play": 0}
        self.snapshot_factories = {}
        self.last_losses = {}
        if checkpoint:
            self._restore(checkpoint)
        self.tensorboard = SummaryWriter(str(self.model_dir / "tensorboard")) if SummaryWriter else None
        self.log_path = self.model_dir / "training_log.tsv"
        if not self.log_path.exists():
            self._write_log_header()

    @staticmethod
    def _device(value):
        if value == "auto":
            value = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(value)
        if device.type == "cuda" and not torch.cuda.is_available():
            return torch.device("cpu")
        return device

    def _restore(self, checkpoint):
        if checkpoint["model_config"] != self.model.model_config():
            raise ValueError("Checkpoint model configuration does not match this run")
        if checkpoint["pool_metadata"] != self.pool.to_dict():
            raise ValueError("Pool JSON and checkpoint pool metadata differ")
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.global_step = int(checkpoint["global_step"])
        self.episode = int(checkpoint["episode"])
        self.update_step = int(checkpoint["update_step"])
        schedule = checkpoint["schedule_state"]
        self.schedule_index = int(schedule["schedule_index"])
        self.next_pool_eval = int(schedule["next_pool_eval"])
        self.next_snapshot = int(schedule["next_snapshot"])
        self.source_selected.update(schedule.get("source_selected", {}))
        restore_random_states(checkpoint["random_states"], self.rng)

    def _write_log_header(self):
        fields = [
            "global_step", "episode", "update_step", "domain", "case",
            "agreement_rate", "agent_utility", "mean_length", "pool_size",
            "opponent1_id", "opponent2_id", "opponent1_source", "opponent2_source",
            "scripted_selected", "snapshot_selected", "self_play_selected",
            "policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction", "event",
        ]
        with self.log_path.open("w", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=fields, delimiter="\t").writeheader()

    def _append_log(self, values):
        fields = [
            "global_step", "episode", "update_step", "domain", "case",
            "agreement_rate", "agent_utility", "mean_length", "pool_size",
            "opponent1_id", "opponent2_id", "opponent1_source", "opponent2_source",
            "scripted_selected", "snapshot_selected", "self_play_selected",
            "policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction", "event",
        ]
        row = {field: values.get(field, "") for field in fields}
        with self.log_path.open("a", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=fields, delimiter="\t").writerow(row)

    def _new_worker(self, domain, case, seed):
        env = AOPEnv(
            domain=domain,
            scale=self.config["scale"],
            case=case,
            add_noise=self.config["opponent_noise"],
        )
        entries = self.pool.sample_pair(
            self.rng, current_model_available=self.config["self_play_probability"] > 0
        )
        for entry in entries:
            self.source_selected[entry.kind] += 1
        factories = []
        for entry in entries:
            if entry.kind == "snapshot":
                factory = self.snapshot_factories.get(entry.id)
                if factory is None:
                    factory = factory_for_entry(entry, self.model_dir, self.model, self.device)
                    self.snapshot_factories[entry.id] = factory
            else:
                factory = factory_for_entry(entry, self.model_dir, self.model, self.device)
            factories.append(factory)
        observation, _ = env.reset(
            seed=seed, options={"opponent_factory": factories, "case": case}
        )
        return {"env": env, "entries": entries, "observation": observation, "episode_start": True}

    def _record_episode(self, worker, info):
        for entry in worker["entries"]:
            if entry.kind != "self_play":
                entry.record(info["agreement"], info["agent_utility"], info["length"])
        self.episode += 1
        self._append_log({
            "global_step": self.global_step,
            "episode": self.episode,
            "update_step": self.update_step,
            "domain": info["domain"],
            "case": info["case"],
            "agreement_rate": float(info["agreement"]),
            "agent_utility": info["agent_utility"],
            "mean_length": info["length"],
            "pool_size": len(self.pool.entries),
            "opponent1_id": worker["entries"][0].id,
            "opponent2_id": worker["entries"][1].id,
            "opponent1_source": worker["entries"][0].kind,
            "opponent2_source": worker["entries"][1].kind,
            "scripted_selected": self.source_selected["scripted"],
            "snapshot_selected": self.source_selected["snapshot"],
            "self_play_selected": self.source_selected["self_play"],
            **self.last_losses,
            "event": "episode",
        })
        if self.tensorboard:
            self.tensorboard.add_scalar("rollout/agreement", float(info["agreement"]), self.global_step)
            self.tensorboard.add_scalar("rollout/agent_utility", info["agent_utility"], self.global_step)
            self.tensorboard.add_scalar("rollout/length", info["length"], self.global_step)

    def collect_rollout(self, domain, case, rollout_steps, n_envs):
        workers = [
            self._new_worker(domain, case, self.config["seed"] + self.episode + index)
            for index in range(n_envs)
        ]
        env0 = workers[0]["env"]
        self.model.eval()
        self.model.set_domain(domain, env0.issues, env0.action_space)
        buffer = RolloutBuffer(
            rollout_steps, n_envs, env0.observation_space, env0.action_space, self.device,
            self.config["gamma"], self.config["gae_lambda"],
        )
        dones = np.zeros(n_envs, dtype=bool)
        try:
            for _ in range(rollout_steps):
                observations = np.stack([worker["observation"] for worker in workers])
                starts = np.asarray([worker["episode_start"] for worker in workers], dtype=np.float32)
                with torch.no_grad():
                    actions, values, log_probs = self.model.sample(
                        torch.as_tensor(observations, device=self.device, dtype=torch.float32)
                    )
                action_values = actions.detach().cpu().numpy()
                rewards = np.zeros(n_envs, dtype=np.float32)
                next_observations = []
                dones = np.zeros(n_envs, dtype=bool)
                for index, (worker, action) in enumerate(zip(workers, action_values)):
                    observation, reward, terminated, truncated, info = worker["env"].step(int(action))
                    done = bool(terminated or truncated)
                    rewards[index] = reward
                    dones[index] = done
                    if done:
                        self._record_episode(worker, info)
                        worker["env"].close()
                        replacement = self._new_worker(
                            domain, case, self.config["seed"] + self.episode + index
                        )
                        workers[index] = replacement
                        observation = replacement["observation"]
                    else:
                        worker["observation"] = observation
                        worker["episode_start"] = False
                    next_observations.append(observation)
                buffer.add(observations, action_values, rewards, starts, values, log_probs)
                self.global_step += n_envs
            final_observations = np.stack(next_observations)
            with torch.no_grad():
                final_values = self.model.predict_values(
                    torch.as_tensor(final_observations, device=self.device, dtype=torch.float32)
                )
            buffer.compute_returns_and_advantage(final_values, dones)
            return buffer
        finally:
            for worker in workers:
                worker["env"].close()

    def update(self, buffer):
        # Keep Transformer dropout disabled so the PPO likelihood ratio uses
        # the same policy definition as rollout collection.  eval() does not
        # disable gradients.
        self.model.eval()
        totals = defaultdict(float)
        batches = 0
        for _ in range(self.config["n_epochs"]):
            for batch in buffer.get(self.config["batch_size"]):
                values, log_prob, entropy = self.model.evaluate_actions(batch.observations, batch.actions)
                values = values.flatten()
                advantages = batch.advantages
                if self.config["normalize_advantage"] and advantages.numel() > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                log_ratio = log_prob - batch.old_log_prob
                ratio = torch.exp(log_ratio)
                loss1 = advantages * ratio
                loss2 = advantages * torch.clamp(
                    ratio, 1.0 - self.config["clip_range"], 1.0 + self.config["clip_range"]
                )
                policy_loss = -torch.min(loss1, loss2).mean()
                value_loss = F.mse_loss(batch.returns, values)
                entropy_mean = entropy.mean()
                loss = (
                    policy_loss
                    + self.config["vf_coef"] * value_loss
                    - self.config["entropy_coef"] * entropy_mean
                )
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config["max_grad_norm"])
                self.optimizer.step()
                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - log_ratio).mean()
                    clip_fraction = (torch.abs(ratio - 1.0) > self.config["clip_range"]).float().mean()
                for key, value in {
                    "policy_loss": policy_loss,
                    "value_loss": value_loss,
                    "entropy": entropy_mean,
                    "approx_kl": approx_kl,
                    "clip_fraction": clip_fraction,
                }.items():
                    totals[key] += float(value.detach().cpu())
                batches += 1
        self.update_step += 1
        self.last_losses = {key: value / max(batches, 1) for key, value in totals.items()}
        if self.tensorboard:
            for key, value in self.last_losses.items():
                self.tensorboard.add_scalar(f"train/{key}", value, self.global_step)
        return self.last_losses

    def _write_evaluation(self, rows):
        directory = self.model_dir / "evaluation"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"step-{self.global_step}.tsv"
        converted = []
        for row in rows:
            my_util = float(row["agent_utility"])
            opp1 = float(row["opponent1_utility"])
            opp2 = float(row["opponent2_utility"])
            converted.append({
                "my_util": my_util,
                "opp_util1": opp1,
                "opp_util2": opp2,
                "social": my_util + opp1 + opp2,
                "nash": my_util * opp1 * opp2,
                "agreement": row["state"]["agreement"],
                "step": row["length"],
                "style": "neutral",
                "opponent1_id": row["opponent1_id"],
                "opponent2_id": row["opponent2_id"],
                "opponent1_source": row["opponent1_source"],
                "opponent2_source": row["opponent2_source"],
                "negotiation_score": negotiation_score(
                    row["agreement"], my_util, row["length"],
                    self.config["score_length_weight"],
                ),
                "seed": row["evaluation_seed"],
                "domain": row["domain"],
                "case": row["case"],
            })
        write_results(path, converted)

    def _write_pool_stats(self):
        path = self.model_dir / "pool" / "stats.tsv"
        exists = path.is_file()
        entries, probabilities = self.pool._flattened_distribution(
            current_model_available=self.config["self_play_probability"] > 0
        )
        fields = (
            "global_step", "opponent_id", "kind", "selected", "matches",
            "agreement_rate", "agent_utility", "mean_length", "difficulty", "probability",
        )
        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
            if not exists:
                writer.writeheader()
            for entry, probability in zip(entries, probabilities):
                writer.writerow({
                    "global_step": self.global_step,
                    "opponent_id": entry.id,
                    "kind": entry.kind,
                    "selected": entry.selected,
                    "matches": entry.matches,
                    "agreement_rate": entry.agreement_rate,
                    "agent_utility": entry.agent_utility,
                    "mean_length": entry.mean_length,
                    "difficulty": entry.difficulty,
                    "probability": float(probability),
                })

    def _pool_event(self, snapshot_due):
        measurements, benchmark, rows = evaluate_pool(
            self.model,
            self.pool,
            self.config["issues"],
            self.config["utility_cases"],
            self.config["pool_eval_episodes"],
            self.model_dir,
            self.config["seed"],
            self.config["score_length_weight"],
            self.global_step,
            self.device,
        )
        event = "pool_evaluation"
        if snapshot_due:
            incumbents = [entry for entry in self.pool.entries if entry.kind == "snapshot"]
            incumbent = max(
                incumbents,
                key=lambda entry: (entry.snapshot_agent_utility, entry.snapshot_score),
                default=None,
            )
            incumbent_metrics = {
                "agent_utility": incumbent.snapshot_agent_utility,
                "negotiation_score": incumbent.snapshot_score,
            } if incumbent else {"agent_utility": float("-inf"), "negotiation_score": float("-inf")}
            accepted = (
                benchmark["agreement_rate"] >= self.config["min_agreement_rate"]
                and (incumbent is None or is_dominant(
                    benchmark,
                    incumbent_metrics,
                    self.config["dominance_tolerance"],
                    self.config["min_agreement_rate"],
                ))
            )
            if accepted:
                path = save_snapshot(self.model_dir, self.model, self.config, self.global_step, benchmark)
                entry = self.pool.add_snapshot(self.global_step, path, benchmark)
                removed = self.pool.prune(protected_ids={entry.id})
                event = f"snapshot_accepted:{entry.id}"
                if removed:
                    event += ";pruned:" + ",".join(removed)
                    for entry_id in removed:
                        self.snapshot_factories.pop(entry_id, None)
            else:
                event = "snapshot_rejected:dominance_or_agreement"
        self.pool.save()
        self._write_evaluation(rows)
        self._write_pool_stats()
        self._append_log({
            "global_step": self.global_step,
            "episode": self.episode,
            "update_step": self.update_step,
            "agreement_rate": benchmark["agreement_rate"],
            "agent_utility": benchmark["agent_utility"],
            "mean_length": benchmark["mean_length"],
            "pool_size": len(self.pool.entries),
            "scripted_selected": self.source_selected["scripted"],
            "snapshot_selected": self.source_selected["snapshot"],
            "self_play_selected": self.source_selected["self_play"],
            **self.last_losses,
            "event": event,
        })
        if self.tensorboard:
            self.tensorboard.add_scalar("pool/size", len(self.pool.entries), self.global_step)
            for entry in self.pool.entries:
                self.tensorboard.add_scalar(f"pool/difficulty/{entry.id}", entry.difficulty, self.global_step)
        return measurements, benchmark

    def _save(self):
        return save_checkpoint(
            self.model_dir,
            self.model,
            self.optimizer,
            self.pool,
            self.config,
            self.global_step,
            self.episode,
            self.update_step,
            self.rng,
            {
                "schedule_index": self.schedule_index,
                "next_pool_eval": self.next_pool_eval,
                "next_snapshot": self.next_snapshot,
                "source_selected": self.source_selected,
            },
        )

    def train(self, total_timesteps):
        target = int(total_timesteps)
        if target < self.global_step:
            raise ValueError(
                f"total_timesteps={target} is below resumed step {self.global_step}; it is a cumulative target"
            )
        schedule = list(
            (domain, case)
            for case in self.config["utility_cases"]
            for domain in self.config["issues"]
        )
        while self.global_step < target:
            domain, case = schedule[self.schedule_index % len(schedule)]
            event_target = min(target, self.next_pool_eval, self.next_snapshot)
            if event_target <= self.global_step:
                event_target = target
            remaining = event_target - self.global_step
            active_envs = min(self.config["num_envs"], remaining)
            rollout_steps = min(self.config["horizon"], max(1, remaining // active_envs))
            if rollout_steps * active_envs > remaining:
                active_envs = 1
                rollout_steps = min(self.config["horizon"], remaining)
            buffer = self.collect_rollout(domain, case, rollout_steps, active_envs)
            self.update(buffer)
            self.schedule_index += 1
            pool_due = self.global_step >= self.next_pool_eval
            snapshot_due = self.global_step >= self.next_snapshot
            if pool_due or snapshot_due:
                self._pool_event(snapshot_due)
                while self.next_pool_eval <= self.global_step:
                    self.next_pool_eval += self.config["pool_eval_freq"]
                while self.next_snapshot <= self.global_step:
                    self.next_snapshot += self.config["snapshot_freq"]
            self._save()
        if self.tensorboard:
            self.tensorboard.flush()
            self.tensorboard.close()
        return self.model_dir
