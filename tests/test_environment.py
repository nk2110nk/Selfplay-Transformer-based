import numpy as np
import pytest

from envs.env import AOPEnv


def test_action_contract_turn_boundary_and_observer_reset():
    env = AOPEnv("Laptop", test=True, add_noise=False)
    observation, _ = env.reset(seed=0, options={"opponent": ["Boulware", "Linear"]})
    assert env.action_space.n == len(env.all_bids) + 1
    assert env._next_actor_is_learner()
    assert not env.observer.has_offer
    with pytest.raises(ValueError, match="Accept"):
        env.step(0)
    observation, _, terminated, _, _ = env.step(1)
    assert terminated or env._next_actor_is_learner()
    assert env.observer.has_offer
    assert not np.shares_memory(env.observer.observation, env.observer.init_observation)
    reset_observation, _ = env.reset(seed=1, options={"opponent": ["Boulware", "Linear"]})
    assert np.all(reset_observation == 0)
    assert not env.observer.has_offer
    assert env.observer.last_valid_step is None
    env.close()


@pytest.mark.parametrize(
    ("case", "order"),
    (("case1", (0, 1, 2)), ("case2", (1, 2, 0)), ("case3", (2, 0, 1))),
)
def test_case_utility_order(case, order):
    env = AOPEnv("Laptop", test=True, add_noise=False, case=case)
    assert env.my_util is env.ufuns[order[0]]
    assert env.opp_util1 is env.ufuns[order[1]]
    assert env.opp_util2 is env.ufuns[order[2]]
    env.close()


def test_case4_is_rejected():
    with pytest.raises(ValueError, match="Unknown utility case"):
        AOPEnv("Laptop", test=True, add_noise=False, case="case4")


def test_training_reward_uses_the_case_learner_utility():
    env = AOPEnv("Laptop", test=False, add_noise=False, case="case2")
    bid = env.all_bids[0]
    agreement = {issue.name: value for issue, value in zip(env.issues, bid)}
    env.state = {"agreement": agreement}
    assert env.get_reward() == pytest.approx(float(env.ufuns[1](bid)))
    env.close()
