import torch

from envs.env import AOPEnv
from policy import Transformer_Policy
from selfplay.snapshot_policy import CurrentPolicyFactory


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
