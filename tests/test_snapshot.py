import torch
import pytest

from envs.env import AOPEnv
from policy import Transformer_Policy
from selfplay.snapshot_policy import CurrentPolicyFactory, build_policy


def test_current_self_play_is_frozen_and_independent():
    env = AOPEnv("Laptop", test=True, add_noise=False)
    model = Transformer_Policy(
        env.observation_space, env.action_space, "cpu", domain=env.domain, issues=env.issues
    )
    factory = CurrentPolicyFactory(model, "cpu", deterministic=True)
    first = factory(env, 0)
    second = factory(env, 1)
    before = [parameter.detach().clone() for parameter in first.model.parameters()]
    with torch.no_grad():
        next(model.parameters()).add_(1.0)
    assert first.observer is not second.observer
    assert first.model is not second.model
    assert all(not parameter.requires_grad for parameter in first.model.parameters())
    for original, frozen in zip(before, first.model.parameters()):
        torch.testing.assert_close(original, frozen)
    env.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cpu_snapshot_state_runs_entirely_on_cuda():
    env = AOPEnv("Laptop", test=True, add_noise=False)
    source = Transformer_Policy(
        env.observation_space, env.action_space, "cpu", domain=env.domain, issues=env.issues
    )
    policy = build_policy(source.model_config(), env, source.state_dict(), "cuda")
    observation, _ = env.reset(options={"opponent": ["Boulware", "Linear"]})
    tensor = torch.as_tensor(observation[None], device="cuda", dtype=torch.float32)
    with torch.no_grad():
        action, _ = policy.mode(tensor)
    assert action.device.type == "cuda"
    assert policy.feature_extractor.memory.device.type == "cuda"
    assert all(parameter.device.type == "cuda" for parameter in policy.parameters())
    env.close()
