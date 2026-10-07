import json
from collections import Counter
import numpy as np

from selfplay.evaluator import balanced_schedule
from selfplay.entry import PoolEntry
from selfplay.pfsp import pfsp_probabilities
from selfplay.pool import OpponentPool, is_dominant


def test_pfsp_prior_and_hard_opponent_priority():
    easy = PoolEntry("easy", "scripted", "Easy", matches=20, agreement_rate=1.0,
                     agent_utility=0.9, mean_length=5, difficulty=0.1)
    hard = PoolEntry("hard", "scripted", "Hard", matches=20, agreement_rate=0.1,
                     agent_utility=0.1, mean_length=75, difficulty=0.9)
    probabilities = pfsp_probabilities([easy, hard], alpha=2.0, uniform_mix=0.1)
    assert np.isfinite(probabilities).all()
    assert (probabilities >= 0).all()
    assert np.isclose(probabilities.sum(), 1.0)
    assert probabilities[1] > probabilities[0]


def test_pool_round_trip_ordered_sampling_and_pruning(tmp_path):
    pool = OpponentPool(tmp_path / "pool", ["Boulware", "Linear"], max_size=4,
                        allow_duplicates=False)
    rng = np.random.default_rng(3)
    pairs = [tuple(entry.id for entry in pool.sample_pair(rng)) for _ in range(50)]
    assert all(first != second for first, second in pairs)
    pool.save()
    loaded = OpponentPool.load(pool.path)
    assert loaded.to_dict() == pool.to_dict()
    for step in (1, 2, 3):
        path = tmp_path / "pool" / "snapshots" / f"snapshot-{step}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"snapshot")
        loaded.add_snapshot(step, path, {"agent_utility": step / 10, "negotiation_score": step})
    removed = loaded.prune(protected_ids={"snapshot-3"})
    assert removed
    assert len(loaded.entries) == loaded.max_size
    assert any(entry.id == "snapshot-3" for entry in loaded.entries)
    assert {entry.name for entry in loaded.entries if entry.kind == "scripted"} == {"Boulware", "Linear"}


def test_snapshot_admission_uses_utility_then_score():
    incumbent = {"agent_utility": 0.6, "negotiation_score": 1.2}
    assert is_dominant({"agreement_rate": 1.0, "agent_utility": 0.7, "negotiation_score": 1.0}, incumbent)
    assert is_dominant({"agreement_rate": 1.0, "agent_utility": 0.605, "negotiation_score": 2.0}, incumbent)
    assert not is_dominant({"agreement_rate": 1.0, "agent_utility": 0.5, "negotiation_score": 3.0}, incumbent)


def test_pool_evaluation_rounds_domains_equally():
    domains = [f"domain{index}" for index in range(7)]
    schedule = balanced_schedule(domains, ["Boulware"], ["case1"], 4)
    assert len(schedule) == 7
    assert Counter(item[0] for item in schedule) == Counter({domain: 1 for domain in domains})

    schedule = balanced_schedule(domains, ["Boulware"], ["case1"], 14)
    assert len(schedule) == 14
    assert Counter(item[0] for item in schedule) == Counter({domain: 2 for domain in domains})

    schedule = balanced_schedule(["Laptop"], ["Boulware"], ["case1"], 4)
    assert len(schedule) == 4
