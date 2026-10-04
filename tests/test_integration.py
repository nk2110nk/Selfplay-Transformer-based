from pathlib import Path

import torch

from selfplay.checkpoint import load_checkpoint
from selfplay.pool import OpponentPool
from train import parse_args, train


def test_training_snapshot_and_cumulative_resume(tmp_path):
    model_dir = tmp_path / "TransformerSelfPlay_Negotiator"
    first = parse_args([
        "-a", "Boulware", "-i", "Laptop", "-sp", str(model_dir),
        "-t", "5", "-n", "4", "-hr", "3", "-bs", "4", "--n-epochs", "1",
        "--pool-eval-freq", "5", "--snapshot-freq", "5", "--pool-eval-episodes", "1",
        "--max-pool-size", "4", "--device", "cpu", "--no-opponent-noise",
    ])
    assert train(first) == model_dir
    checkpoint = load_checkpoint(model_dir)
    assert checkpoint["global_step"] == 5
    pool = OpponentPool.load(model_dir / "pool" / "pool.json")
    assert any(entry.kind == "snapshot" for entry in pool.entries)
    assert list((model_dir / "pool" / "snapshots").glob("snapshot-*.pt"))
    assert (model_dir / "evaluation" / "step-5.tsv").is_file()

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
