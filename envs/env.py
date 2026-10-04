"""Three-party SAOP Gymnasium environment at learner decision boundaries."""

from __future__ import annotations

from typing import Callable, Optional

import gymnasium as gym
import numpy as np

from .inout import load_genius_domain_from_folder
from .observer import EmbeddedObserveHistory
from .rl_negotiator import RLNegotiator
from sao.my_sao import MySAOMechanism
from sao.my_negotiators import (
    AgentGG,
    AgentK,
    Atlas3,
    AverageTitForTatNegotiator,
    HardHeaded,
    TimeBasedNegotiator,
)

PENALTY = -1.0
CASE_ORDERS = {
    "case1": (0, 1, 2),
    "case2": (1, 2, 0),
    "case3": (2, 0, 1),
}


def scripted_opponent(name: str, slot: int = 0, add_noise: bool = True):
    suffix = str(slot)
    if name == "Boulware":
        return TimeBasedNegotiator(name=f"Boulware{suffix}", aspiration_type=10.0, add_noise=add_noise)
    if name == "Linear":
        return TimeBasedNegotiator(name=f"Linear{suffix}", aspiration_type=1.0, add_noise=add_noise)
    if name == "Conceder":
        return TimeBasedNegotiator(name=f"Conceder{suffix}", aspiration_type=0.2, add_noise=add_noise)
    if name == "TitForTat1":
        return AverageTitForTatNegotiator(name=f"TitForTat1{suffix}", gamma=1, add_noise=add_noise)
    if name == "TitForTat2":
        return AverageTitForTatNegotiator(name=f"TitForTat2{suffix}", gamma=2, add_noise=add_noise)
    if name == "AgentK":
        return AgentK(name=f"AgentK{suffix}", add_noise=add_noise)
    if name == "HardHeaded":
        return HardHeaded(name=f"HardHeaded{suffix}", add_noise=add_noise)
    if name == "Atlas3":
        return Atlas3(name=f"Atlas3{suffix}", add_noise=add_noise)
    if name == "AgentGG":
        return AgentGG(name=f"AgentGG{suffix}", add_noise=add_noise)
    raise ValueError(f"Unknown scripted opponent: {name}")


class NaiveEnv(gym.Env):
    metadata = {"render_modes": ["human", "ansi", "rgb_array"]}

    def __init__(self, domain="Laptop", is_first=True, test=False, scale="small",
                 case="case1", add_noise=True, n_steps=80):
        super().__init__()
        scenario = load_genius_domain_from_folder("domain/" + domain)
        if len(scenario.ufuns) < 3:
            raise ValueError(f"{domain} must provide three utility profiles")
        self.domain = domain
        self.issues = scenario.issues
        self.ufuns = tuple(ufun.scale_max(1.0) for ufun in scenario.ufuns[:3])
        self.test = bool(test)
        self.scale = scale
        self.case = case
        self.add_noise = bool(add_noise)
        self.n_steps = int(n_steps)
        self.session: Optional[MySAOMechanism] = None
        self.my_agent: Optional[RLNegotiator] = None
        self.state = None
        self.observer = EmbeddedObserveHistory(domain, self.issues, n_turn=n_steps, scale=scale)
        self._set_utilities(case)
        self.all_bids = self.get_all_bids()
        self.observation_space = self.observer.observation_space
        self.action_space = gym.spaces.Discrete(len(self.all_bids) + 1)
        self.reward_range = (PENALTY, 1.0)
        self.opponent_specs = ("Boulware", "Conceder")

    def _set_utilities(self, case):
        if case not in CASE_ORDERS:
            raise ValueError(f"Unknown utility case: {case}")
        self.case = case
        order = CASE_ORDERS[case]
        self.my_util, self.opp_util1, self.opp_util2 = (self.ufuns[index] for index in order)

    def get_all_bids(self):
        session = MySAOMechanism(issues=self.issues, n_steps=self.n_steps, avoid_ultimatum=False)
        agent = RLNegotiator()
        session.add(agent, ufun=self.my_util)
        return list(agent.all_bids)

    def _build_opponent(self, spec, slot):
        if isinstance(spec, str):
            return scripted_opponent(spec, slot=slot, add_noise=self.add_noise)
        if callable(spec):
            try:
                return spec(self, slot)
            except TypeError:
                return spec()
        raise TypeError(f"Unsupported opponent specification: {spec!r}")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        case = options.get("case", self.case)
        self._set_utilities(case)
        specs = options.get("opponent_factory", options.get("opponent", self.opponent_specs))
        if len(specs) != 2:
            raise ValueError("Three-party negotiation requires exactly two opponents")
        self.opponent_specs = tuple(specs)
        self.session = MySAOMechanism(issues=self.issues, n_steps=self.n_steps, avoid_ultimatum=False)
        self.my_agent = RLNegotiator(name="RLAgent")
        opponents = [self._build_opponent(spec, slot) for slot, spec in enumerate(specs)]
        self.session.add(self.my_agent, ufun=self.my_util)
        self.session.add(opponents[0], ufun=self.opp_util1)
        self.session.add(opponents[1], ufun=self.opp_util2)
        self.state = None
        observation = self.observer.reset()
        if not self._next_actor_is_learner():
            raise RuntimeError("Negotiation did not reset to the learner turn")
        return observation, {"case": self.case, "domain": self.domain}

    def _next_actor_is_learner(self):
        if self.session is None or self.my_agent is None:
            return False
        negotiators = self.session.negotiators
        last = int(getattr(self.session, "_last_checked_negotiator", -1))
        next_index = (last + 1) % len(negotiators)
        return negotiators[next_index].id == self.my_agent.id

    @staticmethod
    def _terminal(state):
        return state["agreement"] is not None or state["timedout"] or state["broken"]

    def step(self, action):
        if isinstance(action, (list, tuple, np.ndarray)):
            action = int(np.asarray(action).reshape(-1)[0])
        action = int(action)
        if not self.action_space.contains(action):
            raise ValueError(f"Action {action} is outside {self.action_space}")
        if not self._next_actor_is_learner():
            raise RuntimeError("Environment step called outside the learner's SAO turn")
        if action == 0:
            if not self.observer.has_offer:
                raise ValueError("Accept is invalid before an opponent offer exists")
            self.my_agent.accept_offer = True
        else:
            self.my_agent.accept_offer = False
            bid = self.all_bids[action - 1]
            self.my_agent.set_next_bid({issue.name: value for issue, value in zip(self.issues, bid)})

        advances = 0
        while True:
            self.state = self.session.step().asdict()
            advances += 1
            observation = self.observer(self.state)
            if self._terminal(self.state):
                info = self.get_info(advances)
                return observation, self.get_reward(), True, False, info
            if self._next_actor_is_learner():
                break
            if advances > len(self.session.negotiators) + 1:
                raise RuntimeError("SAO mechanism did not return to learner turn")
        return observation, 0.0, False, False, {"state": [], "advances": advances}

    def get_reward(self):
        if self.state is None:
            return 0.0
        if self.state["agreement"] is not None:
            offer = tuple(self.state["agreement"].values())
            return float(self.my_util(offer))
        return 0.0 if self.test else PENALTY

    def get_info(self, advances=0):
        agreement = self.state["agreement"] if self.state else None
        if agreement is None:
            utilities = (0.0, 0.0, 0.0)
        else:
            offer = tuple(agreement.values())
            utilities = tuple(float(ufun(offer)) for ufun in (self.my_util, self.opp_util1, self.opp_util2))
        return {
            "state": {
                "step": int(self.state["step"]),
                "agreement": agreement,
                "my_util": utilities[0],
                "opp_util1": utilities[1],
                "opp_util2": utilities[2],
                "last_neg": self.state.get("last_negotiator", ""),
            },
            "agreement": agreement is not None,
            "agent_utility": utilities[0],
            "opponent1_utility": utilities[1],
            "opponent2_utility": utilities[2],
            "social_welfare": sum(utilities),
            "length": int(self.state["step"]),
            "advances": advances,
            "domain": self.domain,
            "case": self.case,
        }

    def close(self):
        self.session = None
        self.my_agent = None


class AOPEnv(NaiveEnv):
    pass
