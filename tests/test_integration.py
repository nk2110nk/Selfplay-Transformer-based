from pathlib import Path
import csv

import pytest
import torch

from envs.env import AOPEnv
from selfplay.checkpoint import load_checkpoint
from selfplay.pool import OpponentPool
from selfplay.snapshot_policy import SnapshotPolicyFactory
from selfplay.trainer import SelfPlayPPOTrainer
from test_negotiator import evaluate
from train import KNOWN_DOMAINS, UNKNOWN_DOMAINS, _new_config, parse_args, train


def test_training_snapshot_and_cumulative_resume(tmp_path):
    model_dir = tmp_path / "TransformerSelfPlay_Negotiator"
    first = parse_args([
        "-a", "Boulware", "-i", "Laptop", "-sp", str(model_dir),
        "-t", "5", "-n", "4", "-hr", "3", "-bs", "4", "--n-epochs", "1",
        "--pool-eval-freq", "5", "--snapshot-freq", "5", "--pool-eval-episodes", "1",
        "--max-pool-size", "4", "--device", "cpu", "--no-opponent-noise",
        "--utility-cases", "case2",
    ])
    assert train(first) == model_dir
    checkpoint = load_checkpoint(model_dir)
    assert checkpoint["global_step"] == 5
    assert checkpoint["utility_cases"] == ["case2"]
    assert checkpoint["utility_orders"] == {"case2": [1, 2, 0]}
    pool = OpponentPool.load(model_dir / "pool" / "pool.json")
    assert any(entry.kind == "snapshot" for entry in pool.entries)
    assert list((model_dir / "pool" / "snapshots").glob("snapshot-*.pt"))
    pool_evaluation = model_dir / "evaluation" / "step-5.tsv"
    assert pool_evaluation.is_file()
    with pool_evaluation.open(newline="", encoding="utf-8") as handle:
        assert {row["case"] for row in csv.DictReader(handle, delimiter="\t")} == {"case2"}

    snapshot_path = next((model_dir / "pool" / "snapshots").glob("snapshot-*.pt"))
    snapshot = torch.load(snapshot_path, map_location="cpu", weights_only=False)
    assert snapshot["utility_cases"] == ["case2"]
    assert snapshot["utility_orders"] == {"case2": [1, 2, 0]}
    incompatible_env = AOPEnv("Laptop", test=True, add_noise=False, case="case1")
    with pytest.raises(ValueError, match="utility case"):
        SnapshotPolicyFactory(snapshot_path)(incompatible_env, 0)
    incompatible_env.close()

    with pytest.raises(ValueError, match="utility_cases"):
        train(parse_args([
            "--resume", str(model_dir), "-t", "6", "--device", "cpu",
            "--utility-cases", "case1",
        ]))
    with pytest.raises(ValueError, match="Checkpoint was trained"):
        evaluate(
            model_dir, ["Boulware", "Boulware"], ["Laptop"], episodes=1,
            seed=0, device="cpu", deterministic=True, case="case1",
        )

    paths = evaluate(
        model_dir, ["Boulware", "Boulware"], ["Laptop"], episodes=1,
        seed=0, device="cpu", deterministic=True, case="case2", style="neutral",
    )
    with paths[0].open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
        assert handle
    assert list(rows[0])[:7] == [
        "my_util", "opp_util1", "opp_util2", "social", "nash", "agreement", "step",
    ]
    assert rows[0]["domain"] == "Laptop"
    assert rows[0]["case"] == "case2"

    resumed = parse_args(["--resume", str(model_dir), "-t", "6", "--device", "cpu"])
    train(resumed)
    checkpoint = load_checkpoint(model_dir)
    assert checkpoint["global_step"] == 6
    assert checkpoint["update_step"] == 3


def test_current_self_play_smoke(tmp_path):
    model_dir = tmp_path / "TransformerSelfPlay_Negotiator"
    args = parse_args([
        "-a", "Boulware", "-i", "Laptop", "-sp", str(model_dir),
        "-t", "1", "-n", "1", "-hr", "1", "-bs", "1", "--n-epochs", "1",
        "--pool-eval-freq", "100", "--snapshot-freq", "100", "--device", "cpu",
        "--self-play-probability", "1", "--scripted-probability", "0",
        "--snapshot-probability", "0", "--no-opponent-noise",
    ])
    train(args)
    assert load_checkpoint(model_dir)["global_step"] == 1


def test_general_model_prevalidates_known_and_unknown_domains(tmp_path):
    args = parse_args([
        "-a", "Boulware", "Conceder", "Linear", "Atlas3",
        "-i", *KNOWN_DOMAINS, "--model-type", "general",
        "--save-path", str(tmp_path / "Transformer_Negotiator"),
        "--total-timesteps", "1", "--device", "cpu",
    ])
    trainer = SelfPlayPPOTrainer(_new_config(args), tmp_path / "Transformer_Negotiator")
    assert set(trainer.config["action_mapping"]) == set(KNOWN_DOMAINS + UNKNOWN_DOMAINS)
    assert all(
        mapping["action_count"] <= 16384
        for mapping in trainer.config["action_mapping"].values()
    )
    if trainer.tensorboard:
        trainer.tensorboard.close()


def test_uninterrupted_and_resumed_updates_match(tmp_path):
    common = [
        "-a", "Boulware", "-i", "Laptop", "-n", "1", "-hr", "1", "-bs", "1",
        "--n-epochs", "1", "--pool-eval-freq", "100", "--snapshot-freq", "100",
        "--device", "cpu", "--no-opponent-noise", "--seed", "17",
    ]
    uninterrupted = tmp_path / "continuous" / "TransformerSelfPlay_Negotiator"
    interrupted = tmp_path / "resumed" / "TransformerSelfPlay_Negotiator"
    train(parse_args([*common, "-sp", str(uninterrupted), "-t", "2"]))
    train(parse_args([*common, "-sp", str(interrupted), "-t", "1"]))
    train(parse_args(["--resume", str(interrupted), "-t", "2", "--device", "cpu"]))
    continuous_state = load_checkpoint(uninterrupted)
    resumed_state = load_checkpoint(interrupted)
    assert continuous_state["schedule_state"] == resumed_state["schedule_state"]
    assert continuous_state["pool_metadata"] == resumed_state["pool_metadata"]
    for name, value in continuous_state["model"].items():
        torch.testing.assert_close(value, resumed_state["model"][name], rtol=0, atol=0)
