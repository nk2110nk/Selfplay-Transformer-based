"""Serializable opponent-pool entries."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math


@dataclass
class PoolEntry:
    id: str
    kind: str
    name: str
    checkpoint_path: str | None = None
    added_step: int = 0
    selected: int = 0
    matches: int = 0
    agreement_rate: float = 0.0
    agent_utility: float = 0.0
    mean_length: float = 0.0
    difficulty: float = 0.5
    snapshot_agent_utility: float = 0.0
    snapshot_score: float = float("-inf")

    def __post_init__(self):
        if self.kind not in {"scripted", "snapshot", "self_play"}:
            raise ValueError(f"Unsupported pool entry kind: {self.kind}")
        if self.matches < 0 or self.selected < 0:
            raise ValueError("Pool counters cannot be negative")

    def record(self, agreement: float, agent_utility: float, length: float):
        values = (agreement, agent_utility, length)
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("Non-finite pool measurement")
        previous = self.matches
        self.matches += 1
        self.agreement_rate = (self.agreement_rate * previous + float(agreement)) / self.matches
        self.agent_utility = (self.agent_utility * previous + float(agent_utility)) / self.matches
        self.mean_length = (self.mean_length * previous + float(length)) / self.matches
        raw = (
            0.45 * (1.0 - self.agreement_rate)
            + 0.40 * (1.0 - self.agent_utility)
            + 0.15 * min(max(self.mean_length / 80.0, 0.0), 1.0)
        )
        confidence = self.matches / (self.matches + 10.0)
        self.difficulty = confidence * raw + (1.0 - confidence) * 0.5

    def to_dict(self):
        data = asdict(self)
        if not math.isfinite(data["snapshot_score"]):
            data["snapshot_score"] = None
        return data

    @classmethod
    def from_dict(cls, data):
        values = dict(data)
        if values.get("snapshot_score") is None:
            values["snapshot_score"] = float("-inf")
        return cls(**values)
