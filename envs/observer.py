"""Observation builders for Transformer negotiation policies."""

from __future__ import annotations

from abc import ABCMeta, abstractmethod
import os

import gymnasium as gym
import numpy as np

from embedding_model import MyEmbedding

os.environ["TOKENIZERS_PARALLELISM"] = "true"

PAD_VALUE = 1e-6


class AbstractObserve(metaclass=ABCMeta):
    def __init__(self):
        self.observation = None
        self.init_observation = None
        self.observation_space = None

    def reset(self):
        if self.init_observation is None:
            raise RuntimeError("Observer has not been initialized")
        self.observation = self.init_observation.copy()
        return self.observation.copy()

    def __call__(self, state):
        if state is None:
            return self.observation.copy()
        return self.observe(state)

    @abstractmethod
    def observe(self, state):
        raise NotImplementedError


class EmbeddedObserveHistory(AbstractObserve):
    """Fixed-length embedded offer history with safe episode reset."""

    def __init__(self, domain, issues=None, n_turn=80, scale="small"):
        super().__init__()
        self.domain = domain
        self.issues = issues
        self.n_turn = int(n_turn)
        self.scale = scale
        self.emb_dim = 256
        self.embedding_model = MyEmbedding(scale=scale)
        self.i_names = [issue.name for issue in self.issues]
        self.i_emb = self.embedding_model.list_embedding(
            self.i_names,
            save_path=f"./embeddings/openai/i_embs/{scale}/{domain}.json",
        )
        self.embedding_model.make_offer_embedding(
            save_path=f"./embeddings/openai/{scale}/{domain}.json"
        )
        shape = (self.n_turn + 1, len(self.i_emb), self.emb_dim)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=shape, dtype=np.float32
        )
        self.init_observation = np.zeros(shape, dtype=np.float32)
        self.observation = self.init_observation.copy()
        self.padded = False
        self.current_step = -1
        self.has_offer = False
        self.valid_mask = np.zeros(shape[0], dtype=bool)

    def reset(self):
        self.observation = self.init_observation.copy()
        self.padded = False
        self.current_step = -1
        self.has_offer = False
        self.valid_mask = np.zeros(self.observation.shape[0], dtype=bool)
        return self.observation.copy()

    @property
    def last_valid_step(self):
        indices = np.flatnonzero(self.valid_mask)
        return int(indices[-1]) if len(indices) else None

    def observe(self, state):
        step = int(state.get("step", 0))
        step = min(max(step, 0), self.observation.shape[0] - 1)
        offers = state.get("new_offers") or []
        for _, offer in offers:
            if offer is None:
                continue
            values = offer.values() if isinstance(offer, dict) else offer
            embedded = self.embedding_model.offer_embedding(values) + self.i_emb
            self.observation[step, :, :] = np.asarray(embedded, dtype=np.float32)
            self.valid_mask[step] = True
            self.has_offer = True
        self.current_step = step
        future = min(step + 1, self.observation.shape[0])
        if future < self.observation.shape[0]:
            self.observation[future:, :, :] = PAD_VALUE
        self.padded = True
        return self.observation.copy()
