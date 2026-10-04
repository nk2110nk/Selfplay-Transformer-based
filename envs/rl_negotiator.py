"""Learner and frozen-policy SAO negotiators."""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from negmas.common import MechanismState
from negmas.outcomes import Outcome
from negmas.sao import ResponseType, SAONegotiator


class RLNegotiator(SAONegotiator):
    def __init__(self, name="RLAgent", accept_offer=False, **kwargs):
        super().__init__(name=name, **kwargs)
        self.next_bid = None
        self.last_bid = None
        self.accept_offer = accept_offer

    def on_ufun_changed(self):
        super().on_ufun_changed()
        self.next_bid = None
        self.last_bid = None

    @property
    def all_bids(self):
        return self._nmi.discrete_outcomes()

    def respond(self, state: MechanismState, offer: Outcome, source=None):
        return ResponseType.ACCEPT_OFFER if self.accept_offer else ResponseType.REJECT_OFFER

    def propose(self, state: MechanismState) -> Optional[Outcome]:
        self.last_bid = self.next_bid
        return self.next_bid

    def set_next_bid(self, next_bid):
        self.next_bid = next_bid


class PolicyNegotiator(RLNegotiator):
    """Independent observer/model wrapper used by frozen self-play opponents."""

    def __init__(self, model, observer, deterministic=False, name="TransformerOpponent", **kwargs):
        super().__init__(name=name, **kwargs)
        self.model = model
        self.observer = observer
        self.deterministic = deterministic
        self.pending_action = None
        self.last_value = None
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def on_ufun_changed(self):
        super().on_ufun_changed()
        self.observer.reset()
        self.pending_action = None

    def _decide(self, state=None):
        state_dict = state.asdict() if hasattr(state, "asdict") else state
        observation = self.observer(state_dict) if state_dict is not None else self.observer(None)
        with torch.no_grad():
            action, _, value = self.model.predict(observation, deterministic=self.deterministic)
        self.pending_action = int(np.asarray(action).reshape(-1)[0])
        self.last_value = float(np.asarray(value).reshape(-1)[0])
        return self.pending_action

    def respond(self, state: MechanismState, offer: Outcome, source=None):
        action = self._decide(state)
        if action == 0:
            if offer is None:
                raise RuntimeError("Snapshot policy attempted Accept without an offer")
            return ResponseType.ACCEPT_OFFER
        return ResponseType.REJECT_OFFER

    def propose(self, state: MechanismState) -> Optional[Outcome]:
        action = self.pending_action if self.pending_action is not None else self._decide(state)
        if action == 0:
            raise RuntimeError("Accept action cannot be converted to a proposal")
        if action > len(self.all_bids):
            raise ValueError(f"Snapshot action {action} exceeds bid count {len(self.all_bids)}")
        bid = self.all_bids[action - 1]
        self.pending_action = None
        return {issue.name: value for issue, value in zip(self.ami.issues, bid)}
