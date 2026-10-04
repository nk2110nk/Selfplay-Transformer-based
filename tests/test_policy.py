import numpy as np
import torch
from torch import nn

from envs.env import AOPEnv
from policy import PAD_VALUE, Transformer_Policy


def make_model():
    env = AOPEnv("Laptop", test=True, add_noise=False)
    model = Transformer_Policy(
        env.observation_space, env.action_space, "cpu", domain=env.domain, issues=env.issues
    )
    model.eval()
    return env, model


def test_accept_mask_and_log_probability_are_identical():
    env, model = make_model()
    observation, _ = env.reset(options={"opponent": ["Boulware", "Linear"]})
    tensor = torch.as_tensor(observation[None])
    with torch.no_grad():
        action, _, old_log_prob = model.sample(tensor)
        _, new_log_prob, _ = model.evaluate_actions(tensor, action[:, None])
    assert action.item() != 0
    torch.testing.assert_close(old_log_prob, new_log_prob)
    env.close()


def test_pooling_uses_later_turns_and_all_issues_but_not_padding():
    env, model = make_model()
    model.feature_extractor = nn.Identity()
    observation = torch.full((1, *env.observation_space.shape), PAD_VALUE)
    observation[:, 0] = 1.0
    observation[:, 2] = 2.0
    baseline = model._pooled_features(observation)
    changed_issue = observation.clone()
    changed_issue[:, 2, 1] = 5.0
    assert not torch.allclose(baseline, model._pooled_features(changed_issue))
    changed_padding = observation.clone()
    changed_padding[:, 40:] = 0.0
    torch.testing.assert_close(baseline, model._pooled_features(changed_padding))
    changed_turn = observation.clone()
    changed_turn[:, 3] = 4.0
    assert not torch.allclose(baseline, model._pooled_features(changed_turn))
    env.close()
