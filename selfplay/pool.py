"""Persistent opponent pool with source-mixture and ordered slot sampling."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .entry import PoolEntry
from .pfsp import pfsp_probabilities

POOL_FORMAT_VERSION = 1


def is_dominant(current, incumbent, tolerance=0.01, min_agreement=0.0):
    if current["agreement_rate"] < min_agreement:
        return False
    utility_delta = current["agent_utility"] - incumbent.get("agent_utility", 0.0)
    if utility_delta > tolerance:
        return True
    if utility_delta < -tolerance:
        return False
    return current["negotiation_score"] > incumbent.get("negotiation_score", float("-inf"))


def snapshot_pruning_key(entry):
    return (entry.difficulty, entry.selected, entry.added_step)


class OpponentPool:
    def __init__(self, root, scripted_names, max_size=12, pfsp_alpha=2.0,
                 uniform_mix=0.1, allow_duplicates=True,
                 self_play_probability=0.0, scripted_probability=0.5,
                 snapshot_probability=0.5, path=None, model_root=None):
        self.root = Path(root)
        self._path = Path(path) if path is not None else self.root / "pool.json"
        self.model_root = Path(model_root) if model_root is not None else self.root.parent
        self.max_size = int(max_size)
        self.pfsp_alpha = float(pfsp_alpha)
        self.uniform_mix = float(uniform_mix)
        self.allow_duplicates = bool(allow_duplicates)
        self.self_play_probability = float(self_play_probability)
        self.scripted_probability = float(scripted_probability)
        self.snapshot_probability = float(snapshot_probability)
        probabilities = np.asarray([
            self.self_play_probability,
            self.scripted_probability,
            self.snapshot_probability,
        ])
        if np.any(~np.isfinite(probabilities)) or np.any(probabilities < 0) or probabilities.sum() <= 0:
            raise ValueError("Source probabilities must be finite, non-negative, and not all zero")
        self.entries = [
            PoolEntry(id=f"scripted-{name}", kind="scripted", name=name)
            for name in dict.fromkeys(scripted_names)
        ]

    @property
    def path(self):
        return self._path

    def _flattened_distribution(self, current_model_available=False):
        groups = []
        if current_model_available and self.self_play_probability > 0:
            groups.append((
                self.self_play_probability,
                [PoolEntry(id="current-self-play", kind="self_play", name="CurrentTransformer")],
            ))
        scripted = [entry for entry in self.entries if entry.kind == "scripted"]
        if scripted and self.scripted_probability > 0:
            groups.append((self.scripted_probability, scripted))
        snapshots = [entry for entry in self.entries if entry.kind == "snapshot"]
        if snapshots and self.snapshot_probability > 0:
            groups.append((self.snapshot_probability, snapshots))
        source_total = sum(weight for weight, _ in groups)
        if source_total <= 0:
            raise ValueError("No available opponent source")
        entries, weights = [], []
        for source_weight, source_entries in groups:
            within = (
                np.ones(len(source_entries), dtype=np.float64) / len(source_entries)
                if source_entries[0].kind == "self_play"
                else pfsp_probabilities(source_entries, self.pfsp_alpha, self.uniform_mix)
            )
            for entry, probability in zip(source_entries, within):
                entries.append(entry)
                weights.append(source_weight / source_total * float(probability))
        probabilities = np.asarray(weights, dtype=np.float64)
        probabilities /= probabilities.sum()
        return entries, probabilities

    def sample_pair(self, rng, current_model_available=False):
        entries, probabilities = self._flattened_distribution(current_model_available)
        if not self.allow_duplicates and len(entries) < 2:
            raise ValueError("Two distinct opponents are required")
        first_index = int(rng.choice(len(entries), p=probabilities))
        if self.allow_duplicates:
            second_index = int(rng.choice(len(entries), p=probabilities))
        else:
            second_probabilities = probabilities.copy()
            second_probabilities[first_index] = 0.0
            second_probabilities /= second_probabilities.sum()
            second_index = int(rng.choice(len(entries), p=second_probabilities))
        selected = (entries[first_index], entries[second_index])
        for entry in selected:
            if entry.kind != "self_play":
                entry.selected += 1
        return selected

    def add_snapshot(self, step, path, metrics):
        path = Path(path)
        try:
            relative = path.relative_to(self.model_root)
        except ValueError:
            relative = path
        entry = PoolEntry(
            id=f"snapshot-{int(step)}",
            kind="snapshot",
            name=f"TransformerSnapshot{int(step)}",
            checkpoint_path=str(relative),
            added_step=int(step),
            snapshot_agent_utility=float(metrics["agent_utility"]),
            snapshot_score=float(metrics["negotiation_score"]),
        )
        if any(existing.id == entry.id for existing in self.entries):
            raise ValueError(f"Duplicate pool entry: {entry.id}")
        self.entries.append(entry)
        return entry

    def prune(self, protected_ids=()):
        protected = set(protected_ids)
        removed = []
        while len(self.entries) > self.max_size:
            candidates = [
                entry for entry in self.entries
                if entry.kind == "snapshot" and entry.id not in protected
            ]
            if not candidates:
                raise ValueError("Pool cannot prune scripted or protected opponents")
            victim = min(candidates, key=snapshot_pruning_key)
            self.entries.remove(victim)
            if victim.checkpoint_path:
                path = self.model_root / victim.checkpoint_path
                if path.is_file():
                    path.unlink()
            removed.append(victim.id)
        return removed

    def to_dict(self):
        return {
            "format_version": POOL_FORMAT_VERSION,
            "max_size": self.max_size,
            "pfsp_alpha": self.pfsp_alpha,
            "uniform_mix": self.uniform_mix,
            "allow_duplicates": self.allow_duplicates,
            "source_probabilities": {
                "self_play": self.self_play_probability,
                "scripted": self.scripted_probability,
                "snapshot": self.snapshot_probability,
            },
            "entries": [entry.to_dict() for entry in self.entries],
        }

    def save(self, path=None):
        destination = Path(path) if path else self.path
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        temporary.replace(destination)

    @classmethod
    def load(cls, path, model_root=None):
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("format_version") != POOL_FORMAT_VERSION:
            raise ValueError("Unsupported pool format")
        source = data["source_probabilities"]
        pool = cls(
            root=path.parent,
            scripted_names=[],
            max_size=data["max_size"],
            pfsp_alpha=data["pfsp_alpha"],
            uniform_mix=data["uniform_mix"],
            allow_duplicates=data["allow_duplicates"],
            self_play_probability=source["self_play"],
            scripted_probability=source["scripted"],
            snapshot_probability=source["snapshot"],
            path=path,
            model_root=model_root,
        )
        pool.entries = [PoolEntry.from_dict(entry) for entry in data["entries"]]
        for entry in pool.entries:
            if entry.kind == "snapshot" and not (pool.model_root / entry.checkpoint_path).is_file():
                raise FileNotFoundError(f"Missing snapshot: {entry.checkpoint_path}")
        return pool
