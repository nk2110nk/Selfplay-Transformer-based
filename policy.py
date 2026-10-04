"""Transformer actor-critic and a compact PPO rollout buffer."""

from __future__ import annotations

from typing import NamedTuple

from gymnasium import spaces
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from NegTransformer import Config, NegTransformer
from embedding_model import MyEmbedding

MAX_ACTIONS = 16384
PAD_VALUE = 1e-6


def get_action_dim(action_space) -> int:
    if isinstance(action_space, spaces.Discrete):
        return 1
    if isinstance(action_space, spaces.MultiDiscrete):
        return len(action_space.nvec)
    if isinstance(action_space, spaces.Box):
        return int(np.prod(action_space.shape))
    raise NotImplementedError(f"Unsupported action space: {action_space}")


def get_obs_shape(observation_space):
    if isinstance(observation_space, tuple):
        return observation_space
    if isinstance(observation_space, spaces.Box):
        return observation_space.shape
    raise NotImplementedError(f"Unsupported observation space: {observation_space}")


class PolicyNet(nn.Module):
    def __init__(self, obs_space, action_space, device):
        super().__init__()
        self.device = torch.device(device)
        self.observation_space = obs_space
        self.action_space = action_space

    def predict(self, observation, state=None, deterministic=False):
        obs = torch.as_tensor(
            np.asarray(observation).reshape(-1, *self.observation_space.shape),
            device=self.device,
            dtype=torch.float32,
        )
        with torch.no_grad():
            actions, values = self._predict(obs, deterministic=deterministic)
        return actions.cpu().numpy().reshape(-1), state, values.cpu().numpy().reshape(-1)


class RolloutBufferSamples(NamedTuple):
    observations: torch.Tensor
    actions: torch.Tensor
    old_values: torch.Tensor
    old_log_prob: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor


class RolloutBuffer:
    def __init__(self, buffer_size, n_envs, obs_space, action_space, device, gamma=0.99, gae_lambda=0.95):
        self.buffer_size = int(buffer_size)
        self.n_envs = int(n_envs)
        self.observation_space = obs_space
        self.obs_dim = get_obs_shape(obs_space)
        self.action_dim = get_action_dim(action_space)
        self.device = torch.device(device)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.reset()

    def reset(self):
        shape = (self.buffer_size, self.n_envs)
        self.observations = np.zeros((*shape, *self.obs_dim), dtype=np.float32)
        self.actions = np.zeros((*shape, self.action_dim), dtype=np.int64)
        self.rewards = np.zeros(shape, dtype=np.float32)
        self.returns = np.zeros(shape, dtype=np.float32)
        self.episode_starts = np.zeros(shape, dtype=np.float32)
        self.values = np.zeros(shape, dtype=np.float32)
        self.log_probs = np.zeros(shape, dtype=np.float32)
        self.advantages = np.zeros(shape, dtype=np.float32)
        self.pos = 0
        self.generator_ready = False

    def add(self, obs, action, reward, episode_start, value, log_prob):
        if self.pos >= self.buffer_size:
            raise RuntimeError("Rollout buffer is full")
        self.observations[self.pos] = obs
        self.actions[self.pos] = np.asarray(action).reshape(self.n_envs, self.action_dim)
        self.rewards[self.pos] = reward
        self.episode_starts[self.pos] = episode_start
        self.values[self.pos] = value.detach().cpu().reshape(-1).numpy()
        self.log_probs[self.pos] = log_prob.detach().cpu().reshape(-1).numpy()
        self.pos += 1

    def compute_returns_and_advantage(self, last_values, dones):
        if self.pos != self.buffer_size:
            raise RuntimeError("Rollout buffer must be full before computing GAE")
        next_values = last_values.detach().cpu().reshape(-1).numpy()
        last_gae = np.zeros(self.n_envs, dtype=np.float32)
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - np.asarray(dones, dtype=np.float32)
                following_values = next_values
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                following_values = self.values[step + 1]
            delta = self.rewards[step] + self.gamma * following_values * next_non_terminal - self.values[step]
            last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            self.advantages[step] = last_gae
        self.returns = self.advantages + self.values

    @staticmethod
    def _flatten(array):
        shape = array.shape
        if len(shape) < 3:
            shape = (*shape, 1)
        return array.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])

    def get(self, batch_size):
        if not self.generator_ready:
            for name in ("observations", "actions", "values", "log_probs", "advantages", "returns"):
                setattr(self, name, self._flatten(getattr(self, name)))
            self.generator_ready = True
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        for start in range(0, len(indices), batch_size):
            batch = indices[start:start + batch_size]
            yield RolloutBufferSamples(*(
                torch.as_tensor(value[batch], device=self.device)
                for value in (
                    self.observations,
                    self.actions,
                    self.values.flatten(),
                    self.log_probs.flatten(),
                    self.advantages.flatten(),
                    self.returns.flatten(),
                )
            ))


class Transformer_Policy(PolicyNet):
    """Shared Transformer policy using valid history and every issue."""

    ARCHITECTURE_VERSION = 2

    def __init__(self, obs_space, action_space, device="cpu", config=None, domain=None,
                 issues=None, is_acceptable=True, decoder_only=False, scale="small",
                 decoder_num=1):
        super().__init__(obs_space, action_space, device)
        self.scale = scale
        self.decoder_only = bool(decoder_only)
        self.decoder_num = int(decoder_num)
        self.is_acceptable = bool(is_acceptable)
        self.n_embd = get_obs_shape(obs_space)[-1]
        self.embedding_model = MyEmbedding(scale=scale)
        self.action_dim = int(action_space.n)
        self.domain = domain
        self.issues = issues
        self._validate_action_dim(self.action_dim)
        self.embedding_model.make_offer_embedding(save_path=f"./embeddings/openai/{scale}/{domain}.json")
        memory = self.embedding_model.outcomes_embedding(
            issues, save_path=f"./embeddings/openai/i_embs/{scale}/{domain}.json"
        )
        if config is None:
            config = Config(action_dim=MAX_ACTIONS, block_size=obs_space.shape[0],
                            n_embd=self.n_embd, decoder_num=self.decoder_num)
        self.feature_extractor = NegTransformer(
            config=config,
            memory=torch.as_tensor(memory, device=self.device, dtype=torch.float32),
            device=self.device,
            decoder_only=self.decoder_only,
        )
        self.fc_p = nn.Linear(config.n_embd, MAX_ACTIONS)
        self.fc_v = nn.Linear(config.n_embd, 1)

    @staticmethod
    def _validate_action_dim(action_dim):
        if not 2 <= int(action_dim) <= MAX_ACTIONS:
            raise ValueError(f"Action count {action_dim} is outside [2, {MAX_ACTIONS}]")

    @staticmethod
    def valid_turn_mask(observation):
        x = torch.as_tensor(observation)
        flattened = x.flatten(start_dim=-2)
        zero = torch.all(
            torch.isclose(flattened, torch.zeros((), dtype=x.dtype, device=x.device)), dim=-1
        )
        pad = torch.all(
            torch.isclose(flattened, torch.full((), PAD_VALUE, dtype=x.dtype, device=x.device)), dim=-1
        )
        return ~(zero | pad)

    def _pooled_features(self, observation):
        raw = torch.as_tensor(observation, device=self.device, dtype=torch.float32)
        encoded = self.feature_extractor(raw)
        valid = self.valid_turn_mask(raw)
        positions = torch.arange(raw.shape[1], device=self.device).unsqueeze(0).expand_as(valid)
        latest = torch.where(valid, positions, torch.zeros_like(positions)).max(dim=1).values
        selected = encoded[torch.arange(raw.shape[0], device=self.device), latest]
        return selected.mean(dim=1)

    def forward(self, observation):
        pooled = self._pooled_features(observation)
        return self.fc_p(pooled)[:, :self.action_dim], self.fc_v(pooled)

    def _masked_log_probs(self, observation, logits):
        if logits.shape[-1] != self.action_dim:
            raise ValueError("Policy/action-space size mismatch")
        masked = logits.clone()
        has_offer = self.valid_turn_mask(observation).any(dim=1).to(masked.device)
        if self.is_acceptable:
            masked[~has_offer, 0] = float("-inf")
        else:
            masked[:, 0] = float("-inf")
        return F.log_softmax(masked, dim=-1)

    def sample(self, observation):
        logits, values = self.forward(observation)
        log_probs = self._masked_log_probs(observation, logits)
        distribution = torch.distributions.Categorical(logits=log_probs)
        actions = distribution.sample()
        return actions, values, distribution.log_prob(actions)

    def mode(self, observation):
        logits, values = self.forward(observation)
        actions = self._masked_log_probs(observation, logits).argmax(dim=-1)
        return actions, values

    def evaluate_actions(self, observation, actions):
        logits, values = self.forward(observation)
        log_probs = self._masked_log_probs(observation, logits)
        distribution = torch.distributions.Categorical(logits=log_probs)
        actions = actions.long().flatten()
        return values, distribution.log_prob(actions), distribution.entropy()

    def predict_values(self, observation):
        return self.fc_v(self._pooled_features(observation))

    def _predict(self, observation, deterministic=False):
        return self.mode(observation) if deterministic else self.sample(observation)[:2]

    def set_domain(self, domain, issues, action_space):
        action_dim = int(action_space.n)
        self._validate_action_dim(action_dim)
        self.domain = domain
        self.issues = issues
        self.action_space = action_space
        self.action_dim = action_dim
        self.embedding_model.make_offer_embedding(save_path=f"./embeddings/openai/{self.scale}/{domain}.json")
        memory = self.embedding_model.outcomes_embedding(
            issues, save_path=f"./embeddings/openai/i_embs/{self.scale}/{domain}.json"
        )
        self.feature_extractor.memory = torch.as_tensor(memory, device=self.device, dtype=torch.float32)
        self.feature_extractor.memory_encoded = None
        self.feature_extractor.isEncoded = False

    def model_config(self):
        return {
            "architecture_version": self.ARCHITECTURE_VERSION,
            "scale": self.scale,
            "decoder_only": self.decoder_only,
            "decoder_num": self.decoder_num,
            "max_actions": MAX_ACTIONS,
            "embedding_dim": self.n_embd,
            "observation_shape": list(self.observation_space.shape),
        }
