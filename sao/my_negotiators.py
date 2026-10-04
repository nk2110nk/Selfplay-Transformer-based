from negmas.sao import SAONegotiator, AspirationNegotiator
from negmas.common import *
from negmas.outcomes import Outcome
from negmas.sao import ResponseType
from negmas.negotiators import Controller
from typing import Optional
from negmas.utilities import UtilityFunction
from sao.opponent_model import HardHeadedFrequencyModel
import numpy as np
import math
import random
from copy import copy


def reset(self):
    del self._nmi


class TimeBasedNegotiator(AspirationNegotiator):
    def __init__(self, add_noise: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.ordered_utils = None
        self.n_outcomes = None
        self.add_noise = add_noise
        self.ordered_outcomes = None

    def on_preferences_changed(self, changes):
        # print("ufun changed")
        super().on_preferences_changed(changes)
        outcomes = self._nmi.discrete_outcomes()
        self.ordered_outcomes = sorted(
            [(self._preferences(outcome), outcome) for outcome in outcomes],
            key=lambda x: x[0],
            reverse=True,
        )
        # print(len(self.ordered_outcomes))
        self.ordered_utils = np.array([u for (u, _) in self.ordered_outcomes[::-1]])
        self.n_outcomes = len(self.ordered_utils)

    def respond(self, state: MechanismState, offer: "Outcome", source=None) -> "ResponseType":
        # if state.step < 0:
        #     # disrupt agreement at the beginning of the negotiation
        #     return ResponseType.REJECT_OFFER
        if isinstance(offer, dict):
            offer = tuple(v for _, v in offer.items())
        if self._preferences is None:
            return ResponseType.REJECT_OFFER
        offered_util = self._preferences(offer)
        if offered_util is None:
            return ResponseType.REJECT_OFFER
        proposal = self.propose(state)
        if proposal is None:
            return ResponseType.REJECT_OFFER
        my_util = self._preferences(tuple(v for _, v in proposal.items()))
        if offered_util >= my_util and (self.reserved_value is None or offered_util > self.reserved_value):
            return ResponseType.ACCEPT_OFFER
        if self.reserved_value is not None and my_util < self.reserved_value:
            return ResponseType.END_NEGOTIATION
        return ResponseType.REJECT_OFFER

    def propose(self, state: MechanismState) -> Optional["Outcome"]:
        asp = (
            self._offering_curve.utility_at(state.relative_time) * (self.ufun_max - self.ufun_min)
            + self.ufun_min
        )
        # ノイズsigma=0.005
        if self.add_noise:
            asp += np.random.normal(0.0, 0.005)
        if self.reserved_value is not None and asp < self.reserved_value:
            return None
        
        # print(self.ordered_utils)
        offer = self.ordered_outcomes[max(0, self.n_outcomes - np.searchsorted(self.ordered_utils, asp) - 1)][1]
        offer = {issue.name: offer[i] for i, issue in enumerate(self.ami.issues)}
        # print(offer)
        return offer


class AverageTitForTatNegotiator(SAONegotiator):
    def __init__(
        self,
        name: str = None,
        parent: Controller = None,
        ufun: Optional["UtilityFunction"] = None,
        gamma: int = 1,
        add_noise: bool = False
    ):
        self.received_utilities = []
        self.proposed_utility = None
        self.ordered_outcomes = None
        self.ordered_utils = None
        self.sent_offer_index = None
        self.n_sent = 0
        self.n_outcomes = None
        self.gamma = gamma
        self.minUtil = 0.0
        self.maxUtil = 1.0
        self.my_last_proposal_utility = 0.0
        self.add_noise = add_noise
        super().__init__(name=name, ufun=ufun, parent=parent)

    def on_preferences_changed(self, changes):
        super().on_preferences_changed(changes)
        outcomes = self._nmi.discrete_outcomes()
        self.ordered_outcomes = sorted(
            [(self._preferences(outcome), outcome) for outcome in outcomes],
            key=lambda x: x[0],
            reverse=True,
        )
        self.ordered_utils = np.array([u for (u, _) in self.ordered_outcomes[::-1]])
        self.n_outcomes = len(self.ordered_utils)
        self.minUtil = self.ordered_outcomes[-1][0]
        self.maxUtil = self.ordered_outcomes[0][0]

    def respond(self, state: MechanismState, offer: "Outcome", source=None) -> "ResponseType":
        if self._preferences is None:
            return ResponseType.REJECT_OFFER
        offered_util = self._preferences(tuple(v for _, v in offer.items()))
        if offered_util is None:
            return ResponseType.REJECT_OFFER
        if len(self.received_utilities) >= self.gamma + 1:
            self.received_utilities = self.received_utilities[1:]
        self.received_utilities.append(offered_util)
        n_sent = self.n_sent
        proposal = self.propose(state)
        if proposal is None:
            return ResponseType.REJECT_OFFER
        my_util = self._preferences(tuple(v for _, v in proposal.items()))
        self.my_last_proposal_utility = my_util
        self.n_sent = n_sent
        if offered_util >= my_util and (self.reserved_value is None or offered_util > self.reserved_value):
            return ResponseType.ACCEPT_OFFER
        if self.reserved_value is not None and my_util < self.reserved_value:
            return ResponseType.END_NEGOTIATION
        return ResponseType.REJECT_OFFER

    def _get_bid_index(self, ulevel: float) -> int:
        return max(0, self.n_outcomes - np.searchsorted(self.ordered_utils, ulevel) - 1)

    def average_tit_for_tat(self) -> float:
        opponent_last_util = self.received_utilities[-1]
        opponent_first_util = self.received_utilities[0]
        try:
            relative_change = opponent_first_util / opponent_last_util
        except ZeroDivisionError:
            relative_change = float('inf')
        my_last_util = self.my_last_proposal_utility
        target_util = relative_change * my_last_util
        return min(max(target_util, self.minUtil), self.maxUtil)

    def propose(self, state: MechanismState) -> Optional[Outcome]:
        if len(self.received_utilities) > self.gamma:
            target = self.average_tit_for_tat()
            # ノイズsigma=0.005
            if self.add_noise:
                target += np.random.normal(0.0, 0.005)
            bid = self.ordered_outcomes[self._get_bid_index(target)][1]
        else:
            bid = self.ordered_outcomes[self.n_sent][1]
            self.n_sent += 1
        offer = {issue.name: bid[i] for i, issue in enumerate(self.ami.issues)}
        return offer


class AgentK(SAONegotiator):
    def __init__(
        self,
        name: str = 'AgentK',
        parent: Controller = None,
        ufun: Optional["UtilityFunction"] = None,
        add_noise: bool = False
    ):
        self.offered_bid_map = list()
        self.target = 1.0
        self.bid_target = 1.0
        self.sum = 0.0
        self.sum2 = 0.0
        self.rounds = 0
        self.tremor = 2.0
        self.add_noise = add_noise

        self.ordered_outcomes = None
        self.ordered_utils = None
        self.n_outcomes = None
        super().__init__(name=name, ufun=ufun, parent=parent)

    def on_preferences_changed(self, changes):
        super().on_preferences_changed(changes)
        outcomes = self._nmi.discrete_outcomes()
        self.ordered_outcomes = sorted(
            [(self._preferences(outcome), outcome) for outcome in outcomes],
            key=lambda x: x[0],
            reverse=True,
        )
        self.ordered_utils = np.array([u for (u, _) in self.ordered_outcomes[::-1]])
        self.n_outcomes = len(self.ordered_utils)

    def respond(self, state: MechanismState, offer: "Outcome", source=None) -> "ResponseType":
        # if isinstance(offer, dict):
        #     offer = tuple(v for _, v in offer.items())
        if self._preferences is None:
            return ResponseType.REJECT_OFFER
        try:
            p = self.accept_probability(state, offer)
        except ZeroDivisionError:
            return ResponseType.ACCEPT_OFFER
        if p > np.random.rand():
            return ResponseType.ACCEPT_OFFER
        return ResponseType.REJECT_OFFER

    def accept_probability(self, state: MechanismState, offer: "Outcome") -> float:
        offer_tuple = tuple(v for _, v in offer.items())
        offered_util = self._preferences(offer_tuple)
        if offered_util is None:
            return 0.0
        self.offered_bid_map.append((offer, offered_util))
        self.sum += offered_util
        self.sum2 += offered_util * offered_util
        self.rounds += 1
        mean = self.sum / self.rounds
        variance = (self.sum2 / self.rounds) - (mean * mean)
        if variance < 0.0:
            variance = 0.0
        deviation = math.sqrt(variance * 12)
        if math.isnan(deviation):
            deviation = 0.0
        time = state.relative_time
        t = time ** 3
        if offered_util > 1.0:
            offered_util = 1.0
        concede = 0.1

        estimate_max = mean + ((1 - mean) * deviation)
        alpha = 1 + self.tremor + (10 * mean) - (2 * self.tremor * mean)
        beta = alpha + (np.random.rand() * self.tremor) - (self.tremor / 2)
        pre_target = 1 - (math.pow(time, alpha) * (1 - estimate_max))
        pre_target2 = 1 - (math.pow(time, beta) * (1 - estimate_max))
        if 1 - pre_target < 1e-12:
            pre_target = 1 - 1e-12
        if 1 - pre_target2 < 1e-12:
            pre_target2 = 1 - 1e-12
        ratio = (deviation + concede) / (1. - pre_target)
        if math.isnan(ratio) or ratio > 2.0:
            ratio = 2.0
        # print(ratio)
        ratio2 = (deviation + concede) / (1. - pre_target2)
        if math.isnan(ratio2) or ratio2 > 2.0:
            ratio2 = 2.0
        self.target = ratio * pre_target + 1 - ratio
        self.bid_target = ratio2 * pre_target2 + 1 - ratio2

        m = t * (-300) + 400
        if self.target > estimate_max:
            r = self.target - estimate_max
            f = 1 / (r * r)
            if f > m or math.isnan(f):
                f = m
            app = r * f / m
            self.target = self.target - app
        else:
            self.target = estimate_max

        if self.bid_target > estimate_max:
            r = self.bid_target - estimate_max
            f = 1 / (r * r)
            if f > m or math.isnan(f):
                f = m
            app = r * f / m
            self.bid_target = self.bid_target - app
        else:
            self.bid_target = estimate_max

        utility_evaluation = offered_util - estimate_max
        satisfy = offered_util - self.target

        # print(self.target, offered_util)
        p = (math.pow(time, 5) / 5) + utility_evaluation + satisfy
        if p < 0.1:
            p = 0.0
        # print(p)
        return p

    def search_bid(self):
        return {issue.name: random.choice(issue.values) for issue in self.ami.issues}

    def _get_bid_index(self, ulevel: float) -> int:
        return max(0, self.n_outcomes - np.searchsorted(self.ordered_utils, ulevel) - 1)

    def propose(self, state: MechanismState) -> Optional[Outcome]:
        bid_temp = []
        next_bid = None
        noise = np.random.normal(0.0, 0.005) if self.add_noise else 0

        for bid, util in self.offered_bid_map:
            if util > self.target:
                bid_temp.append(bid)

        size = len(bid_temp)
        if size > 0:
            # next_bid = bid_temp[np.random.randint(size)]
            next_bid = random.choice(bid_temp)
        else:
            search_util = 0.0
            loop = 0
            while search_util < self.bid_target + noise:
                if loop > 500:
                    self.bid_target -= 0.01
                    loop = 0
                next_bid = self.search_bid()
                # print(next_bid)
                next_bid_tuple = tuple([v for k, v in next_bid.items()])
                search_util = self.ufun(next_bid_tuple)
                loop += 1
        return next_bid


class HardHeaded(SAONegotiator):
    TOP_SELECTED_BIDS = 4
    LEARNING_COEF = 0.2
    LEARNING_VALUE_ADDITION = 1
    UTILITY_TOLORANCE = 0.01

    def __init__(
        self,
        name: str = 'HardHeaded',
        parent: Controller = None,
        ufun: Optional["UtilityFunction"] = None,
        add_noise: bool = False
    ):
        self.Ka = 0.05
        self.e = 0.05
        self.discountF = 1.
        self.lowest_yet_utility = 1.
        self.offer_queue = list()
        self.opponent_last_bid = None
        self.first_round = True
        self.number_of_issue = 0
        self.min_util = 0.0
        self.max_util = 1.0
        self.my_last_proposal_utility = 0.0
        self.opponent_best_bid = None
        self.next_offer = None
        self.round = 0
        self.opponent_model = None
        self.bid_history = []
        self.my_history = []
        self.add_noise = add_noise

        self.ordered_outcomes = None
        self.ordered_utils = None
        self.n_outcomes = None
        super().__init__(name=name, ufun=ufun, parent=parent)

    def on_preferences_changed(self, changes):
        super().on_preferences_changed(changes)
        outcomes = self._nmi.discrete_outcomes()
        self.ordered_outcomes = sorted(
            [(self._preferences(outcome), outcome) for outcome in outcomes],
            key=lambda x: x[0],
            reverse=True,
        )
        self.ordered_utils = np.array([u for (u, _) in self.ordered_outcomes[::-1]])
        self.n_outcomes = len(self.ordered_utils)
        self.min_util = self.reserved_value
        self.max_util = self.ordered_outcomes[0][0]
        self.number_of_issue = len(self.ami.issues)

        self.opponent_model = HardHeadedFrequencyModel(
            self._preferences,
            learn_coef=self.LEARNING_COEF,
            learn_value_addition=self.LEARNING_VALUE_ADDITION,
        )

    def respond(self, state: MechanismState, offer: "Outcome", source=None) -> "ResponseType":
        offer_tuple = tuple(v for _, v in offer.items())
        if self._preferences is None:
            return ResponseType.REJECT_OFFER

        self.opponent_last_bid = offer_tuple
        self.bid_history.append(offer_tuple)
        self.opponent_model.update(offer, state.relative_time)
        proposal = self.propose(state)
        if proposal is None:
            return ResponseType.REJECT_OFFER
        my_util = self._preferences(tuple(v for _, v in proposal.items()))
        self.my_last_proposal_utility = my_util

        if self.opponent_best_bid is None:
            self.opponent_best_bid = offer_tuple
        elif self._preferences(offer_tuple) > self._preferences(self.opponent_best_bid):
            self.opponent_best_bid = offer_tuple

        self.round += 1
        p = self.get_p(state)

        if self.first_round:
            self.first_round = not self.first_round
            new_bid = self.ordered_outcomes[0][1]
            self.offer_queue.append(new_bid)
        elif not self.offer_queue:
            new_bids = list()
            util, new_bid = self.ordered_outcomes[self._get_bid_index(self.my_last_proposal_utility) + 1]
            if util >= p:
                new_bids.append(new_bid)
            else:
                new_bids.append(random.choice(self.my_history))
                # new_bids.append(self.my_history[np.random.randint(len(self.my_history))])

            first_util = util
            add_util, add_bid = self.ordered_outcomes[self._get_bid_index(first_util) + 1]
            while first_util - add_util < self.UTILITY_TOLORANCE and add_util >= p:
                new_bids.append(add_bid)
                add_util, add_bid = self.ordered_outcomes[self._get_bid_index(add_util) + 1]
            if len(new_bids) <= self.TOP_SELECTED_BIDS:
                self.offer_queue.extend(new_bids)
            else:
                added_so_far = 0
                while added_so_far <= self.TOP_SELECTED_BIDS:
                    best_bid = new_bids[-1]
                    for e in new_bids:
                        if self.opponent_model(e) > self.opponent_model(best_bid):
                            best_bid = e
                    self.offer_queue.append(best_bid)
                    del new_bids[-1]
                    added_so_far += 1

            if self._preferences(self.offer_queue[0]) < self._preferences(self.opponent_best_bid):
                self.offer_queue.insert(0, self.opponent_best_bid)

        if not self.offer_queue:
            # util, best_bid1 = self.ordered_outcomes[np.random.randint(self.n_outcomes)]
            util, best_bid1 = random.choice(self.ordered_outcomes)
            if self.opponent_last_bid is not None and util <= self._preferences(self.opponent_last_bid):
                new_action = ResponseType.ACCEPT_OFFER
            elif best_bid1 is None:
                new_action = ResponseType.ACCEPT_OFFER
            else:
                new_action = best_bid1
                if util < self.lowest_yet_utility:
                    self.lowest_yet_utility = util

        if self.opponent_last_bid is not None \
                and (self._preferences(self.opponent_last_bid) > self.lowest_yet_utility
                     or self._preferences(self.offer_queue[0]) <= self._preferences(self.opponent_last_bid)):
            new_action = ResponseType.ACCEPT_OFFER
        else:
            new_action = self.offer_queue.pop(0)
            self.bid_history.append(new_action)
            if self._preferences(new_action) < self.lowest_yet_utility:
                self.lowest_yet_utility = self._preferences(new_action)
            # # ノイズsigma=0.005
            # if self.add_noise:
            #     noise = np.random.normal(0.0, 0.005)
            # next_bid = self.ordered_outcomes[self._get_bid_index(self.target + noise)][1]
        if new_action == ResponseType.ACCEPT_OFFER:
            return new_action
        else:
            self.next_offer = new_action
            return ResponseType.REJECT_OFFER

    def _get_bid_index(self, ulevel: float) -> int:
        return max(0, self.n_outcomes - np.searchsorted(self.ordered_utils, ulevel) - 1)

    def get_p(self, state: MechanismState):
        time = state.relative_time
        step_point = self.discountF
        temp_max = self.max_util
        temp_min = self.min_util
        ignore_discount_threshold = 0.9

        if step_point >= ignore_discount_threshold:
            fa = self.Ka + (1 - self.Ka) * math.pow(time / step_point, 1 / self.e)
            p = self.min_util + (1 - fa) * (self.max_util - self.min_util)
        elif time <= step_point:
            temp_e = self.e / step_point
            fa = self.Ka + (1 - self.Ka) * math.pow(time / step_point, 1 / temp_e)
            temp_min += abs(temp_max - temp_min) * step_point
            p = temp_min + (1 - fa) * (temp_max - temp_min)
        else:
            temp_e = 30
            fa = (self.Ka + (1 - self.Ka) * math.pow((time - step_point) / (1 - step_point), 1 / temp_e))
            temp_max = temp_min + abs(temp_max - temp_min) * step_point
            p = temp_min + (1 - fa) * (temp_max - temp_min)
        return p

    def propose(self, state: MechanismState) -> Optional[Outcome]:
        if self.next_offer is None:
            self.first_round = not self.first_round
            new_bid = self.ordered_outcomes[0][1]
            self.my_history.append(new_bid)
            return {issue.name: new_bid[_] for _, issue in enumerate(self.ami.issues)}
        else:
            self.my_history.append(self.next_offer)
            return {issue.name: self.next_offer[_] for _, issue in enumerate(self.ami.issues)}


# class CUHKAgent(SAONegotiator):
#     class OwnBidHistory:
#         def __init__(self):
#             self.BidHistory = list()
#             self.UtilHistory = list()
#             self.minBidInHistory = None
#             self.minUtilInHistory = 1.

#         def add_bid(self, bid, util):
#             if bid not in self.BidHistory:
#                 self.BidHistory.append(bid)
#                 self.UtilHistory.append(util)
#             if util < self.minUtilInHistory:
#                 self.minBidInHistory = bid
#                 self.minUtilInHistory = util

#         def choose_lower_bid_in_history(self):
#             min_utility = 100
#             min_bid = None
#             for bid, util in zip(self.BidHistory, self.UtilHistory):
#                 if util < min_utility:
#                     min_utility = util
#                     min_bid = bid
#             return min_bid

#     class OpponentBidHistory:
#         def __init__(self):
#             self.bid_history = list()
#             self.util_history = list()
#             self.opponent_bids_statistics = dict()
#             self.maximum_bids_stored = 100
#             self.bid_counter = dict()
#             self.bid_maximum_from_opponent = None
#             self.util_maximum_from_opponent = 0.
#             self.issues = None

#         def add_bid(self, bid, util):
#             if bid not in self.bid_history:
#                 self.bid_history.append(bid)
#                 self.util_history.append(util)
#             if util > self.util_maximum_from_opponent:
#                 self.bid_maximum_from_opponent = bid
#                 self.util_maximum_from_opponent = util

#         def initialize_data_structures(self, issues):
#             self.issues = issues
#             for issue in issues:
#                 issue_values_map = dict()
#                 for value in issue.values:
#                     issue_values_map[value] = 0
#                 self.opponent_bids_statistics[issue] = issue_values_map

#         @staticmethod
#         def bid2tuple(bid):
#             return tuple(sorted(bid.items(), key=lambda x: x[0]))

#         def get_concession_degree(self):
#             num_of_bids = len(self.bid_history)
#             num_of_distinct_bid = 0
#             history_length = 10
#             if num_of_bids - history_length > 0:
#                 for j in range(num_of_bids - history_length, num_of_bids):
#                     if self.bid_counter[self.bid2tuple(self.bid_history[j])] == 1:
#                         num_of_distinct_bid += 1
#                 concession_degree = math.pow(num_of_distinct_bid / history_length, 2)
#             else:
#                 num_of_distinct_bid = self.get_size()
#                 concession_degree = math.pow(num_of_distinct_bid / history_length, 2)
#             return concession_degree

#         def get_size(self):
#             num_of_bids = len(self.bid_history)
#             bid_counter = dict()
#             for i in range(num_of_bids):
#                 tuple_bid = self.bid2tuple(self.bid_history[i])
#                 if tuple_bid not in bid_counter:
#                     bid_counter[tuple_bid] = 1
#                 else:
#                     bid_counter[tuple_bid] += 1
#             return len(bid_counter)

#         def get_best_bid_in_history(self):
#             return self.bid_maximum_from_opponent

#         def choose_bid(self, candidate_bids):
#             upper_search_limit = 200
#             max_index = -1
#             max_frequency = 0

#             if len(candidate_bids) >= upper_search_limit:
#                 candidate_bids = np.random.choice(candidate_bids, upper_search_limit)

#             for i in range(len(candidate_bids)):
#                 max_value = 0
#                 index = 0
#                 for issue in self.issues:
#                     v = candidate_bids[i][issue.name]
#                     max_value = self.opponent_bids_statistics[self.issues[index]][v]
#                     index += 1
#                 if max_value > max_frequency:
#                     max_frequency = max_value
#                     max_index = i
#                 elif max_value == max_frequency:
#                     if np.random.rand() < 0.5:
#                         max_frequency = max_value
#                         max_index = i

#             if max_index == -1:
#                 # return candidate_bids[np.random.randint(len(candidate_bids))]
#                 return random.choice(candidate_bids)
#             else:
#                 if np.random.rand() < 0.95:
#                     return candidate_bids[max_index]
#                 else:
#                     # return candidate_bids[np.random.randint(len(candidate_bids))]
#                     return random.choice(candidate_bids)

#         def choose_best_from_history(self):
#             max_util = -1
#             max_bid = None
#             for bid, util in zip(self.bid_history, self.util_history):
#                 if max_util < util:
#                     max_util = util
#                     max_bid = bid
#             return max_bid

#         def update_statistics(self, bid_to_update, to_remove):
#             for issue in self.opponent_bids_statistics:
#                 v = bid_to_update[issue.name]
#                 if to_remove:
#                     self.opponent_bids_statistics[issue][v] -= 1
#                 else:
#                     self.opponent_bids_statistics[issue][v] += 1

#         def update_opponent_model(self, bid_to_update, util):
#             tuple_bid = self.bid2tuple(bid_to_update)
#             self.add_bid(bid_to_update, util)
#             if tuple_bid not in self.bid_counter:
#                 self.bid_counter[tuple_bid] = 1
#             else:
#                 self.bid_counter[tuple_bid] += 1
#             if len(self.bid_history) <= self.maximum_bids_stored:
#                 self.update_statistics(bid_to_update, False)

#     def __init__(
#             self,
#             name: str = 'CUHKAgent',
#             parent: Controller = None,
#             ufun: Optional["UtilityFunction"] = None,
#             add_noise: bool = False
#     ):
#         self.maximumOfBid = None
#         self.ownBidHistory = CUHKAgent.OwnBidHistory()
#         self.opponentBidHistory = CUHKAgent.OpponentBidHistory()
#         self.bidsBetweenUtility = list()
#         self.bid_maximum_utility = None
#         self.utility_threshold = 0.0
#         self.MaximumUtility = 0.0
#         self.timeLeftAfter = 0
#         self.timeLeftBefore = 0
#         self.totalTime = 0
#         self.maximumTimeOfOpponent = 0
#         self.maximumTimeOfOwn = 0
#         self.minConcedeToDiscountingFactor = 0.08
#         self.concedeToDiscountingFactor = None
#         self.concedeToDiscountingFactor_original = None
#         self.discountingFactor = 1.0
#         self.minimumUtilityThreshold = 0.
#         self.choose_concede_to_discounting_degree()

#         self.concedeToOpponent = False
#         self.toughAgent = False
#         self.alpha1 = 2
#         self.reservationValue = 0
#         self.ActionOfOpponent = None
#         self.next_offer = None
#         self.add_noise = add_noise

#         self.ordered_outcomes = None
#         self.ordered_utils = None
#         self.n_outcomes = None
#         super().__init__(name=name, ufun=ufun, parent=parent)

#     def on_preferences_changed(self, changes):
#         super().on_preferences_changed(changes)
#         outcomes = self._nmi.discrete_outcomes()
#         self.ordered_outcomes = sorted(
#             [(self._preferences(outcome), outcome) for outcome in outcomes],
#             key=lambda x: x[0],
#             reverse=True,
#         )
#         self.ordered_utils = np.array([u for (u, _) in self.ordered_outcomes[::-1]])
#         self.n_outcomes = len(self.ordered_utils)
#         self.maximumOfBid = self.n_outcomes
#         self.reservationValue = self.reserved_value
#         self.utility_threshold, self.bid_maximum_utility = self.ordered_outcomes[0]
#         self.MaximumUtility = self.utility_threshold
#         self.calculate_bids_between_utility()
#         self.opponentBidHistory.initialize_data_structures(self.ami.issues)
#         self.totalTime = self.ami.n_steps

#     def calculate_bids_between_utility(self):
#         min_utility = self.minimumUtilityThreshold
#         maximum_rounds = int((self.MaximumUtility - min_utility) / 0.01)
#         for i in range(maximum_rounds):
#             self.bidsBetweenUtility.append(list())
#         self.bidsBetweenUtility[-1].append(self.bid_maximum_utility)
#         limits = 0
#         if self.maximumOfBid < 20000:
#             for u, b in self.ordered_outcomes:
#                 for i in range(maximum_rounds):
#                     if i * 0.01 + min_utility <= u <= (i + 1) * 0.01 + min_utility:
#                         self.bidsBetweenUtility[i].append(b)
#                         break
#         else:
#             while limits <= 20000:
#                 b = self.random_search_bid()
#                 for i in range(maximum_rounds):
#                     if i * 0.01 + min_utility <= self.ufun(b) <= (i + 1) * 0.01 + min_utility:
#                         self.bidsBetweenUtility[i].append(b)
#                         break
#                 limits += 1

#     def random_search_bid(self):
#         return {issue.name: random.choice(issue.values) for issue in self.ami.issues}

#     def choose_concede_to_discounting_degree(self):
#         if self.discountingFactor > 0.75:
#             beta = 1.8
#         elif self.discountingFactor > 0.5:
#             beta = 1.5
#         else:
#             beta = 1.2
#         alpha = math.pow(self.discountingFactor, beta)
#         self.concedeToDiscountingFactor = alpha + (1 - alpha) * self.minConcedeToDiscountingFactor
#         self.concedeToDiscountingFactor_original = self.concedeToDiscountingFactor

#     def update_concede_degree(self):
#         gama = 1
#         weight = 0.1
#         opponent_toughness_degree = self.opponentBidHistory.get_concession_degree()
#         self.concedeToDiscountingFactor = self.concedeToDiscountingFactor_original + weight * (1 - self.concedeToDiscountingFactor_original) * math.pow(opponent_toughness_degree, gama)
#         if self.concedeToDiscountingFactor >= 1:
#             self.concedeToDiscountingFactor = 1

#     def estimate_round_left(self, opponent, state):
#         if opponent:
#             if self.timeLeftBefore - self.timeLeftAfter > self.maximumTimeOfOpponent:
#                 self.maximumTimeOfOpponent = self.timeLeftBefore - self.timeLeftAfter
#         else:
#             if self.timeLeftAfter - self.timeLeftBefore > self.maximumTimeOfOwn:
#                 self.maximumTimeOfOwn = self.timeLeftAfter - self.timeLeftBefore
#         if self.maximumTimeOfOpponent + self.maximumTimeOfOwn != 0:
#             round_left = (self.totalTime - state.step) / (self.maximumTimeOfOpponent + self.maximumTimeOfOwn)
#         else:
#             round_left = self.totalTime - state.step
#         return int(round_left / 2)

#     def bid_to_offer(self, state: MechanismState):
#         decreasing_amount1 = 0.05
#         decreasing_amount2 = 0.25
#         maximum_of_bid = self.MaximumUtility
#         time = state.relative_time
#         if self.discountingFactor == 1 and self.maximumOfBid > 3000:
#             minimum_of_bid = self.MaximumUtility - decreasing_amount1
#             if self.discountingFactor > 1 - decreasing_amount2 and self.maximumOfBid > 10000 and time >= 0.98:
#                 minimum_of_bid = self.MaximumUtility - decreasing_amount2
#             if self.utility_threshold > minimum_of_bid:
#                 self.utility_threshold = minimum_of_bid
#         else:
#             if time <= self.concedeToDiscountingFactor:
#                 min_threshold = (maximum_of_bid * self.discountingFactor) / math.pow(self.discountingFactor, self.concedeToDiscountingFactor)
#                 self.utility_threshold = maximum_of_bid - (maximum_of_bid - min_threshold) * math.pow(time / self.concedeToDiscountingFactor, self.alpha1)
#             else:
#                 self.utility_threshold = (maximum_of_bid * self.discountingFactor) / math.pow(self.discountingFactor, time)
#             minimum_of_bid = self.utility_threshold
#         best_opp_bid = self.opponentBidHistory.get_best_bid_in_history()
#         if self.ufun(best_opp_bid) >= self.utility_threshold or self.ufun(best_opp_bid) >= minimum_of_bid:
#             return best_opp_bid
#         candidate_bids = self.get_bids_between_utility(minimum_of_bid, maximum_of_bid)
#         bid_returned = self.opponentBidHistory.choose_bid(candidate_bids)
#         if bid_returned is None:
#             bid_returned = self.bid_maximum_utility
#         return bid_returned

#     def accept_opponent_offer(self, opponent_bid, own_bid, time):
#         current_utility = self.ufun(opponent_bid)
#         next_round_utility = self.ufun(own_bid)
#         maximum_utility = self.MaximumUtility
#         self.concedeToOpponent = False
#         if current_utility >= self.utility_threshold or current_utility >= next_round_utility:
#             return True
#         else:
#             predict_maximum_utility = maximum_utility * self.discountingFactor
#             current_maximum_utility = self.ufun(self.opponentBidHistory.get_best_bid_in_history()) * math.pow(self.discountingFactor, time)
#             if current_maximum_utility > predict_maximum_utility and time > self.concedeToDiscountingFactor:
#                 if self.ufun(opponent_bid) >= self.ufun(self.opponentBidHistory.get_best_bid_in_history()) - 0.01:
#                     return True
#                 else:
#                     self.concedeToOpponent = True
#                     return False
#             elif current_maximum_utility > self.utility_threshold * math.pow(self.discountingFactor, time):
#                 if self.ufun(opponent_bid) >= self.ufun(self.opponentBidHistory.get_best_bid_in_history()) - 0.01:
#                     return True
#                 else:
#                     self.concedeToOpponent = True
#                     return False
#             else:
#                 return False

#     def terminate_current_negotiation(self, own_bid, time):
#         self.concedeToOpponent = False
#         current_utility = self.reservationValue
#         next_round_utility = self.ufun(own_bid)
#         maximum_utility = self.MaximumUtility
#         if current_utility >= self.utility_threshold or current_utility >= next_round_utility:
#             return True
#         else:
#             predict_maximum_utility = maximum_utility * self.discountingFactor
#             current_maximum_utility = self.reservationValue * math.pow(self.discountingFactor, time)
#             if current_maximum_utility > predict_maximum_utility and time > self.concedeToDiscountingFactor:
#                 return True
#             else:
#                 return False

#     def get_bids_between_utility(self, lower_bound, upper_bound):
#         bids_in_range = list()
#         util_range = int((upper_bound - self.minimumUtilityThreshold) / 0.01)
#         initial = int((lower_bound - self.minimumUtilityThreshold) / 0.01)
#         for i in range(initial, util_range):
#             bids_in_range.extend(self.bidsBetweenUtility[i])
#         if not bids_in_range:
#             bids_in_range.append(self.bid_maximum_utility)
#         return bids_in_range

#     def respond(self, state: MechanismState, offer: "Outcome", source=None) -> "ResponseType":
#         self.ActionOfOpponent = offer
#         if self._preferences is None:
#             return ResponseType.REJECT_OFFER
#         self.timeLeftBefore = state.step
#         self.opponentBidHistory.update_opponent_model(offer, self.ufun(offer))
#         self.update_concede_degree()
#         if len(self.ownBidHistory.BidHistory) == 0:
#             bid = self.bid_maximum_utility
#             action = bid
#         else:
#             if self.estimate_round_left(True, state) > 10:
#                 bid = self.bid_to_offer(state)
#                 is_accept = self.accept_opponent_offer(offer, bid, state.relative_time)
#                 is_terminate = self.terminate_current_negotiation(bid, state.relative_time)
#                 if is_accept and not is_terminate:
#                     action = ResponseType.ACCEPT_OFFER
#                 elif is_terminate and not is_accept:
#                     action = ResponseType.END_NEGOTIATION
#                 elif is_accept and is_terminate:
#                     if self.ufun(offer) > self.reservationValue:
#                         action = ResponseType.ACCEPT_OFFER
#                     else:
#                         action = ResponseType.END_NEGOTIATION
#                 else:
#                     if self.concedeToOpponent:
#                         bid = self.opponentBidHistory.get_best_bid_in_history()
#                         action = bid
#                         self.toughAgent = True
#                         self.concedeToOpponent = False
#                     else:
#                         action = bid
#                         self.toughAgent = False
#             else:
#                 if state.relative_time > 0.9985 and self.estimate_round_left(True, state) < 5:
#                     bid = self.opponentBidHistory.get_best_bid_in_history()
#                     if self.ufun(bid) < 0.85:
#                         candidate_bids = self.get_bids_between_utility(self.MaximumUtility - 0.15, self.MaximumUtility - 0.02)
#                         if self.estimate_round_left(True, state) < 2:
#                             bid = self.opponentBidHistory.get_best_bid_in_history()
#                         else:
#                             bid = self.opponentBidHistory.choose_bid(candidate_bids)
#                         if bid is None:
#                             bid = self.opponentBidHistory.get_best_bid_in_history()
#                     is_accept = self.accept_opponent_offer(offer, bid, state.relative_time)
#                     is_terminate = self.terminate_current_negotiation(bid, state.relative_time)
#                     if is_accept and not is_terminate:
#                         action = ResponseType.ACCEPT_OFFER
#                     elif is_terminate and not is_accept:
#                         action = ResponseType.END_NEGOTIATION
#                     elif is_accept and is_terminate:
#                         if self.ufun(offer) > self.reservationValue:
#                             action = ResponseType.ACCEPT_OFFER
#                         else:
#                             action = ResponseType.END_NEGOTIATION
#                     else:
#                         if self.toughAgent:
#                             action = ResponseType.ACCEPT_OFFER
#                         else:
#                             action = bid
#                 else:
#                     bid = self.bid_to_offer(state)
#                     is_accept = self.accept_opponent_offer(offer, bid, state.relative_time)
#                     is_terminate = self.terminate_current_negotiation(bid, state.relative_time)
#                     if is_accept and not is_terminate:
#                         action = ResponseType.ACCEPT_OFFER
#                     elif is_terminate and not is_accept:
#                         action = ResponseType.END_NEGOTIATION
#                     elif is_accept and is_terminate:
#                         if self.ufun(offer) > self.reservationValue:
#                             action = ResponseType.ACCEPT_OFFER
#                         else:
#                             action = ResponseType.END_NEGOTIATION
#                     else:
#                         action = bid
#         self.ownBidHistory.add_bid(bid, self.ufun(bid))
#         self.timeLeftAfter = state.step
#         self.estimate_round_left(False, state)

#         if action == ResponseType.ACCEPT_OFFER or action == ResponseType.END_NEGOTIATION:
#             return action
#         else:
#             self.next_offer = bid
#             return ResponseType.REJECT_OFFER

#     def _get_bid_index(self, ulevel: float) -> int:
#         return max(0, self.n_outcomes - np.searchsorted(self.ordered_utils, ulevel) - 1)

#     def propose(self, state: MechanismState) -> Optional[Outcome]:
#         if self.ActionOfOpponent is None:
#             return self.bid_maximum_utility
#         return self.next_offer
#         # ノイズsigma=0.005
#         # if self.add_noise:
#         #     noise = np.random.normal(0.0, 0.005)
#         # next_bid = self.ordered_outcomes[self._get_bid_index(self.target + noise)][1]
#         # return next_bid


class Atlas3(SAONegotiator):
    class NegotiationInfo:
        def __init__(self, ufun: "UtilityFunction", ami: "AgentMechanismInterface"):
            self.negotiator_num = len(ami.participants)
            self.ufun = ufun
            self.issues = ami.issues
            self.bou = 0.0
            self.mpbu = 0.0
            self.time_scale = 0.0
            self.round = 0
            self.my_bid_history = list()
            self.opponent_bid_history = list()
            self.opponent_sum = 0.0
            self.opponent_pow_sum = 0.0
            self.opponent_average = 0.0
            self.opponent_variance = 0.0
            self.opponent_standard_deviation = 0.0
            self.bob_history = list()
            self.pblist = list()
            self.value_relative_utility = dict()
            self.all_value_frequency = dict()
            self.opponent_value_frequency = dict()

            self.init_all_value_frequency()
            self.init_value_relative_utility()
            self.init_opponent()

        def init_opponent(self):
            self.init_opponent_value_frequency()

        def init_opponent_value_frequency(self):
            for issue in self.issues:
                self.opponent_value_frequency[issue] = dict()
                for value in issue.values:
                    self.opponent_value_frequency[issue][value] = 0

        def init_all_value_frequency(self):
            for issue in self.issues:
                self.all_value_frequency[issue] = dict()
                for value in issue.values:
                    self.all_value_frequency[issue][value] = 0

        def init_value_relative_utility(self):
            for issue in self.issues:
                self.value_relative_utility[issue] = dict()
                for value in issue.values:
                    self.value_relative_utility[issue][value] = 0.0

        def set_value_relative_utility(self, max_bid):
            for i, issue in enumerate(self.issues):
                current_bid = list(copy(max_bid))
                for value in issue.values:
                    current_bid[i] = value
                    current_bid_util = self.ufun(tuple(current_bid))
                    self.value_relative_utility[issue][value] = current_bid_util - self.ufun(max_bid)

        def get_value_by_frequency_list(self, issue):
            max_f = 0
            max_value = None
            idx = np.random.choice(len(issue.values), len(issue.values), replace=False)
            random_order_value = [issue.values[_] for _ in idx]
            for value in random_order_value:
                current_f = self.opponent_value_frequency[issue][value]
                if max_value is None or current_f > max_f:
                    max_f = current_f
                    max_value = value
            return max_value

        def update_info(self, offered_bid):
            self.update_negotiation_info(offered_bid)
            self.update_frequency_list(offered_bid)

        def update_frequency_list(self, offered_bid):
            for i, issue in enumerate(self.issues):
                value = offered_bid[i]
                self.opponent_value_frequency[issue][value] += 1
                self.all_value_frequency[issue][value] += 1

        def update_negotiation_info(self, offered_bid):
            self.opponent_bid_history.append(offered_bid)
            util = self.ufun(offered_bid)
            self.opponent_sum += util
            self.opponent_pow_sum += math.pow(util, 2)
            round_num = len(self.opponent_bid_history)
            self.opponent_average = self.opponent_sum / round_num
            self.opponent_variance = self.opponent_pow_sum / round_num - math.pow(self.opponent_average, 2)
            if self.opponent_variance < 0:
                self.opponent_variance = 0.0
            self.opponent_standard_deviation = math.sqrt(self.opponent_variance)
            if util > self.bou:
                self.bob_history.append(offered_bid)
                self.bou = util

        def update_time_scale(self, time):
            self.round += 1
            self.time_scale = time / self.round

        def update_my_bid_history(self, bid):
            self.my_bid_history.append(bid)

        def update_pblist(self, popular_bid):
            if popular_bid not in self.pblist:
                self.pblist.append(popular_bid)
                self.mpbu = max(self.mpbu, self.ufun(popular_bid))
                self.pblist.sort(key=self.ufun)

    NEAR_ITERATION = 1

    def __init__(
            self,
            name: str = 'Atlas3',
            parent: Controller = None,
            ufun: Optional["UtilityFunction"] = None,
            add_noise: bool = False
    ):
        self.rv = 0.
        self.df = 1.
        self.a11 = 0.
        self.a12 = 0.
        self.a21 = 0.
        self.a22 = 0.
        self.tf = 1.
        self.pf = .5
        self.offered_bid = None
        self.supporter_num = 0
        self.clist_index = 0
        self.negotiation_info: Optional["Atlas3.NegotiationInfo"] = None
        self.next_offer = None
        self.max_bid = None
        self.max_util = 0.0
        self.add_noise = add_noise

        self.ordered_outcomes = None
        self.ordered_utils = None
        self.n_outcomes = None
        super().__init__(name=name, ufun=ufun, parent=parent)

    def on_preferences_changed(self, changes):
        super().on_preferences_changed(changes)
        outcomes = self._nmi.discrete_outcomes()
        self.ordered_outcomes = sorted(
            [(self._preferences(outcome), outcome) for outcome in outcomes],
            key=lambda x: x[0],
            reverse=True,
        )
        self.ordered_utils = np.array([u for (u, _) in self.ordered_outcomes[::-1]])
        self.rv = self.reserved_value
        self.max_util, self.max_bid = self.ordered_outcomes[0]
        self.negotiation_info = Atlas3.NegotiationInfo(self.ufun, self.ami)
        self.negotiation_info.set_value_relative_utility(self.max_bid)
        self.n_outcomes = len(self.ordered_utils)

    def respond(self, state: MechanismState, offer: "Outcome", source=None) -> "ResponseType":
        offer_tuple = tuple(v for _, v in offer.items())
        self.offered_bid = offer_tuple
        if self._preferences is None:
            return ResponseType.REJECT_OFFER

        self.negotiation_info.update_info(offer_tuple)
        self.negotiation_info.update_pblist(offer_tuple)

        action = self.choose_action(state)
        if action == ResponseType.ACCEPT_OFFER or action == ResponseType.END_NEGOTIATION:
            return action
        else:
            self.next_offer = action
            return ResponseType.REJECT_OFFER

    def choose_action(self, state: MechanismState):
        time = state.relative_time
        self.negotiation_info.update_time_scale(time)
        clist = self.negotiation_info.pblist
        if time > 1.0 - self.negotiation_info.time_scale * (len(clist) + 1):
            return self.choose_final_action(state, clist)

        if self.select_accept(self.offered_bid, time):
            return ResponseType.ACCEPT_OFFER

        if self.select_end_negotiation(time):
            return ResponseType.END_NEGOTIATION
        return self.offer_action(time)

    def choose_final_action(self, state: MechanismState, clist) -> "ResponseType":
        offered_bid_util = 0
        if self.offered_bid is not None:
            offered_bid_util = self.ufun(self.offered_bid)
        if self.clist_index >= len(clist):
            if offered_bid_util >= self.rv:
                return ResponseType.ACCEPT_OFFER
            else:
                return ResponseType.REJECT_OFFER

        cbid = clist[self.clist_index]
        cbid_util = self.ufun(cbid)
        if cbid_util > offered_bid_util and cbid_util > self.rv:
            self.clist_index += 1
            self.negotiation_info.update_my_bid_history(cbid)
            return cbid
        elif offered_bid_util > self.rv:
            return ResponseType.ACCEPT_OFFER
        return self.offer_action(state.relative_time)

    def _get_bid_index(self, ulevel: float) -> int:
        return max(0, self.n_outcomes - np.searchsorted(self.ordered_utils, ulevel) - 1)

    def get_bid(self, base_bid, threshold):
        bid = self.get_bid_neighborhood_search(base_bid, threshold)
        if self.ufun(bid) < threshold:
            bid = self.get_bid_by_appropriate_search(base_bid, threshold)
        if self.ufun(bid) < threshold:
            bid = copy(self.max_bid)
        bid = self.get_convert_bid_by_frequency_list(bid)
        return bid
        # return self.ordered_outcomes[self._get_bid_index(threshold)][1]

    def get_bid_neighborhood_search(self, base_bid, threshold):
        bid = base_bid
        for i in range(self.NEAR_ITERATION):
            bid = self.neighborhood_search(bid, threshold)
        return bid

    def neighborhood_search(self, base_bid, threshold):
        current_bid = list(copy(base_bid))
        target_bids = list()
        target_bid_util = 0.0

        for i, issue in enumerate(self.negotiation_info.issues):
            values = issue.values
            for value in values:
                current_bid[i] = value
                current_bid_util = self.ufun(tuple(current_bid))
                if current_bid_util >= threshold:
                    if not target_bids:
                        target_bids.append(tuple(current_bid))
                        target_bid_util = current_bid_util
                    else:
                        if current_bid_util < target_bid_util:
                            target_bids = list()
                            target_bids.append(tuple(current_bid))
                            target_bid_util = current_bid_util
                        elif current_bid_util == target_bid_util:
                            target_bids.append(tuple(current_bid))
            current_bid = list(copy(base_bid))
        if target_bids:
            # return target_bids[np.random.randint(len(target_bids))]
            return random.choice(target_bids)

        else:
            return base_bid

    def get_bid_by_appropriate_search(self, base_bid, threshold):
        bid = self.relative_utility_search(threshold)
        if self.ufun(bid) < threshold:
            bid = base_bid
        return bid

    def relative_utility_search(self, threshold):
        bid = list(copy(self.max_bid))
        d = threshold - 1.0
        concession_sum = 0.0
        value_relative_utility = self.negotiation_info.value_relative_utility
        idx = np.random.choice(len(self.negotiation_info.issues), len(self.negotiation_info.issues), replace=False)
        random_issues = [self.negotiation_info.issues[_] for _ in idx]
        for i, issue in zip(idx, random_issues):
            j = np.random.choice(len(issue.values), len(issue.values), replace=False)
            random_values = [issue.values[_] for _ in j]
            for value in random_values:
                relative_utility = value_relative_utility[issue][value]
                if d <= concession_sum + relative_utility:
                    bid[i] = value
                    concession_sum += relative_utility
                    break
        return tuple(bid)

    def get_convert_bid_by_frequency_list(self, base_bid):
        current_bid = copy(base_bid)
        idx = np.random.choice(len(self.negotiation_info.issues), len(self.negotiation_info.issues), replace=False)
        random_issues = [self.negotiation_info.issues[_] for _ in idx]
        for i, issue in zip(idx, random_issues):
            next_bid = list(copy(current_bid))
            next_bid[i] = self.negotiation_info.get_value_by_frequency_list(issue)
            if self.ufun(tuple(next_bid)) >= self.ufun(current_bid):
                current_bid = next_bid
        return tuple(current_bid)

    def propose(self, state: MechanismState) -> Optional[Outcome]:
        if self.next_offer is None:
            new_bid = self.offer_action(state.relative_time)
            return new_bid
        else:
            return self.next_offer

    def offer_action(self, time):
        noise = 0
        # ノイズsigma=0.005
        if self.add_noise:
            noise = np.random.normal(0.0, 0.005)
        # offer_bid = self.get_bid(self.ami.outcomes[np.random.randint(self.n_outcomes)], self.get_threshold(time) + noise)
        offer_bid = self.get_bid(random.choice(self.ami.outcomes), self.get_threshold(time) + noise)
        # offer_bid = self.ordered_outcomes[self._get_bid_index(self.get_threshold(time) + noise)][1]
        self.negotiation_info.update_my_bid_history(offer_bid)
        offer_bid = {issue.name: offer_bid[i] for i, issue in enumerate(self.ami.issues)}
        return offer_bid

    def select_accept(self, offered_bid, time):
        offered_bid_util = self.ufun(offered_bid)
        if offered_bid_util >= self.get_threshold(time):
            return True
        else:
            return False

    def select_end_negotiation(self, time):
        if self.rv * math.pow(self.df, time) >= self.get_threshold(time):
            return True
        else:
            return False

    def get_threshold(self, time):
        self.update_game_matrix()
        target = self.get_expected_utility_in_fop() / math.pow(self.df, time)
        if self.df == 1.0:
            threshold = target + (1.0 - target) * (1.0 - time)
        else:
            threshold = max(1.0 - time / self.df, target)
        return threshold

    def get_expected_utility_in_fop(self):
        q = self.get_opponent_ees()
        return q * self.a21 + (1 - q) * self.a22

    def get_opponent_ees(self):
        q = 1.0
        if (self.a12 - self.a22 != 0) and (1.0 - (self.a11 - self.a21) / (self.a12 - self.a22) != 0):
            q = 1.0 / (1.0 - (self.a11 - self.a21) / (self.a12 - self.a22))
        if q < 0.0 or q > 1.0:
            q = 1.0
        return q

    def update_game_matrix(self):
        if self.negotiation_info.negotiator_num == 2:
            c = self.negotiation_info.bou
        else:
            c = self.negotiation_info.mpbu

        self.a11 = self.rv * self.df
        self.a12 = math.pow(self.df, self.tf)
        if c >= self.rv:
            self.a21 = c * math.pow(self.df, self.tf)
        else:
            self.a21 = self.rv * self.df
        self.a21 = self.pf * self.a21 + (1.0 - self.pf) * self.a12


class AgentGG(SAONegotiator):
    class ImpMap(dict):
        def __init__(self, ami: "AgentMechanismInterface"):
            super().__init__()
            self.issues = ami.issues
            for issue in self.issues:
                self[issue.name] = [AgentGG.ImpUnit(value) for value in issue.values]
                # self[issue.name] = {value: AgentGG.ImpUnit(value) for value in issue.values}
                # self[issue.name] = OrderedDict([(value, AgentGG.ImpUnit(value)) for value in issue.values])
                # issue_imp_unit = list()
                # number_in_issue = len(issue.values)
                # for i in range(number_in_issue):
                #     issue_imp_unit.append(AgentGG.ImpUnit(issue.values[i]))
                # self[issue.name] = issue_imp_unit

        def opponent_update(self, received_offer_bid: "Outcome"):
            for issue in received_offer_bid:
                # self[issue][received_offer_bid[issue]].mean_weight_sum += 1
                current_issue_list = self[issue]
                for current_unit in current_issue_list:
                    if current_unit.value_of_issue == received_offer_bid[issue]:
                        current_unit.mean_weight_sum += 1
                        break
            for issue, imp_unit_list in self.items():
                # self[issue] = OrderedDict(sorted(imp_unit_list.items(), key=lambda x: x.mean_weight_sum, reverse=True))
                imp_unit_list.sort(key=lambda x: x.mean_weight_sum, reverse=True)

        def self_update(self, ordered_outcomes):
            current_weight = 0
            for _, bid in ordered_outcomes[::-1]:
                current_weight += 1
                for idx, issue in enumerate(self.issues):
                    # current_unit = self[issue][bid[issue]]
                    # current_unit.weight_sum += current_weight
                    # current_unit.count += 1
                    current_issue_list = self[issue.name]
                    for current_unit in current_issue_list:
                        if current_unit.value_of_issue == bid[idx]:
                            current_unit.weight_sum += current_weight
                            current_unit.count += 1
                            break

            for imp_unit_list in self.values():
                for current_unit in imp_unit_list:
                    if current_unit.count == 0:
                        current_unit.mean_weight_sum = 0.0
                    else:
                        current_unit.mean_weight_sum = current_unit.weight_sum / current_unit.count

            for imp_unit_list in self.values():
                imp_unit_list.sort(key=lambda x: x.mean_weight_sum, reverse=True)

            min_mean_weight_sum = np.inf
            for k, v in self.items():
                temp_mean_weight_sum = v[-1].mean_weight_sum
                if temp_mean_weight_sum < min_mean_weight_sum:
                    min_mean_weight_sum = temp_mean_weight_sum

            for imp_unit_list in self.values():
                for current_unit in imp_unit_list:
                    current_unit.mean_weight_sum -= min_mean_weight_sum

        def get_importance(self, bid):
            if isinstance(bid, tuple):
                bid = {i.name: bid[idx] for idx, i in enumerate(self.issues)}
            return sum([i.mean_weight_sum for issue, value in bid.items() for i in self[issue] if i.value_of_issue == value])
            # bid_importance = 0.0
            # for issue, value in bid.items():
            #     value_importance = 0.0
            #     for i in self[issue]:
            #         if i.value_of_issue == value:
            #             value_importance = i.mean_weight_sum
            #             break
            #     bid_importance += value_importance
            # return bid_importance

    class ImpUnit:
        def __init__(self, value):
            self.value_of_issue = value
            self.weight_sum = 0
            self.count = 0
            self.mean_weight_sum = 0.0

    def __init__(
            self,
            name: str = 'AgentGG',
            parent: Controller = None,
            ufun: Optional["UtilityFunction"] = None,
            add_noise: bool = False
    ):
        self.offer_lower_ratio = 1.0
        self.offer_higher_ratio = 1.1
        self.imp_map = None
        self.opponent_imp_map = None
        self.MAX_IMPORTANCE = None
        self.MIN_IMPORTANCE = None
        self.MEDIAN_IMPORTANCE = None
        self.MAX_IMPORTANCE_BID = None
        self.MIN_IMPORTANCE_BID = None
        self.OPPONENT_MAX_IMPORTANCE = None
        self.OPPONENT_MIN_IMPORTANCE = None
        self.receivedBid = None
        self.initialOpponentBid = None
        self.lastBidValue = None
        self.reservationImportanceRatio = None
        self.offerRandomly = True
        self.startTime = None
        self.maxOppoBidImpForMeGot = False
        self.maxOppoBidImpForMe = 0.0
        self.estimatedNashPoint = None
        self.lastReceivedBid = None
        self.initialTimePass = False
        self.add_noise = add_noise

        self.ordered_outcomes_only = None
        self.ordered_outcomes = None
        self.ordered_utils = None
        self.n_outcomes = None
        super().__init__(name=name, ufun=ufun, parent=parent)

    def on_preferences_changed(self, changes):
        super().on_preferences_changed(changes)
        outcomes = self._nmi.discrete_outcomes()
        self.ordered_outcomes = sorted(
            [(self._preferences(outcome), outcome) for outcome in outcomes],
            key=lambda x: x[0],
            reverse=True,
        )
        self.ordered_outcomes_only = [i[1] for i in self.ordered_outcomes]
        self.ordered_utils = np.array([u for (u, _) in self.ordered_outcomes[::-1]])
        self.n_outcomes = len(self.ordered_utils)
        self.imp_map = AgentGG.ImpMap(self.ami)
        self.opponent_imp_map = AgentGG.ImpMap(self.ami)
        self.imp_map.self_update(self.ordered_outcomes)
        self.get_max_and_min_bid()
        self.get_median_bid()
        self.reservationImportanceRatio = self.get_reservation_ratio()

    def get_max_and_min_bid(self):
        l_values1 = dict()
        l_values2 = dict()
        for k, v in self.imp_map.items():
            l_values1[k] = v[0].value_of_issue
            l_values2[k] = v[-1].value_of_issue
        self.MAX_IMPORTANCE_BID = l_values1
        self.MIN_IMPORTANCE_BID = l_values2
        self.MAX_IMPORTANCE = self.imp_map.get_importance(l_values1)
        self.MIN_IMPORTANCE = self.imp_map.get_importance(l_values2)

    def get_median_bid(self):
        median = int(self.n_outcomes / 2)
        self.MEDIAN_IMPORTANCE = self.imp_map.get_importance(self.ordered_outcomes[median][1])
        if self.n_outcomes % 2 == 0:
            self.MEDIAN_IMPORTANCE += self.imp_map.get_importance(self.ordered_outcomes[median + 1][1])
            self.MEDIAN_IMPORTANCE /= 2

    def get_reservation_ratio(self):
        median_bid_ratio = (self.MEDIAN_IMPORTANCE - self.MIN_IMPORTANCE) / (self.MAX_IMPORTANCE - self.MIN_IMPORTANCE)
        return self.reserved_value * median_bid_ratio / 0.5

    def get_max_oppo_bid_imp_for_me(self, time, time_last):
        this_bid_imp = self.imp_map.get_importance(self.receivedBid)
        if this_bid_imp > self.maxOppoBidImpForMe:
            self.maxOppoBidImpForMe = this_bid_imp
        if self.initialTimePass:
            if time - self.startTime > time_last:
                max_oppo_bid_ratio_for_me = (self.maxOppoBidImpForMe - self.MIN_IMPORTANCE) / (self.MAX_IMPORTANCE - self.MIN_IMPORTANCE)
                self.estimatedNashPoint = (1 - max_oppo_bid_ratio_for_me) / 1.7 + max_oppo_bid_ratio_for_me
                self.maxOppoBidImpForMeGot = True
        else:
            if self.lastReceivedBid != self.receivedBid:
                self.initialTimePass = True
                self.startTime = time

    def get_threshold(self, time):
        if time < 0.01:
            self.offer_lower_ratio = 0.9999
        elif time < 0.02:
            self.offer_lower_ratio = 0.99
        elif time < 0.2:
            self.offer_lower_ratio = 0.99 - 0.5 * (time - 0.02)
        elif time < 0.5:
            self.offerRandomly = False
            p2 = 0.3 * (1 - self.estimatedNashPoint) + self.estimatedNashPoint
            self.offer_lower_ratio = 0.9 - (0.9 - p2) / (0.5 - 0.2) * (time - 0.2)
        elif time < 0.9:
            p1 = 0.3 * (1 - self.estimatedNashPoint) + self.estimatedNashPoint
            p2 = 0.15 * (1 - self.estimatedNashPoint) + self.estimatedNashPoint
            self.offer_lower_ratio = p1 - (p1 - p2) / (0.9 - 0.5) * (time - 0.5)
        elif time < 0.98:
            p1 = 0.15 * (1 - self.estimatedNashPoint) + self.estimatedNashPoint
            p2 = 0.05 * (1 - self.estimatedNashPoint) + self.estimatedNashPoint
            possible_ratio = p1 - (p1 - p2) / (0.98 - 0.9) * (time - 0.9)
            self.offer_lower_ratio = max(possible_ratio, self.reservationImportanceRatio + 0.3)
        elif time < 0.995:
            p1 = 0.05 * (1 - self.estimatedNashPoint) + self.estimatedNashPoint
            p2 = 0.0 * (1 - self.estimatedNashPoint) + self.estimatedNashPoint
            possible_ratio = p1 - (p1 - p2) / (0.995 - 0.98) * (time - 0.98)
            self.offer_lower_ratio = max(possible_ratio, self.reservationImportanceRatio + 0.25)
        elif time < 0.999:
            p1 = 0.0 * (1 - self.estimatedNashPoint) + self.estimatedNashPoint
            p2 = -0.35 * (1 - self.estimatedNashPoint) + self.estimatedNashPoint
            possible_ratio = p1 - (p1 - p2) / (0.9989 - 0.995) * (time - 0.995)
            self.offer_lower_ratio = max(possible_ratio, self.reservationImportanceRatio + 0.25)
        else:
            possible_ratio = -0.4 * (1 - self.estimatedNashPoint) + self.estimatedNashPoint
            self.offer_lower_ratio = max(possible_ratio, self.reservationImportanceRatio + 0.2)
        self.offer_higher_ratio = self.offer_lower_ratio + 0.1

    def respond(self, state: MechanismState, offer: "Outcome", source=None) -> "ResponseType":
        self.receivedBid = offer
        if self._preferences is None:
            return ResponseType.REJECT_OFFER
        time = state.relative_time
        imp_ratio_for_me = (self.imp_map.get_importance(offer) - self.MIN_IMPORTANCE) / (self.MAX_IMPORTANCE - self.MIN_IMPORTANCE)
        if imp_ratio_for_me >= self.offer_lower_ratio:
            return ResponseType.ACCEPT_OFFER
        if not self.maxOppoBidImpForMeGot:
            self.get_max_oppo_bid_imp_for_me(time, 3.0 / 1000.0)
        if time < 0.3:
            self.opponent_imp_map.opponent_update(offer)
        self.get_threshold(time)
        if time >= 0.9989:
            ratio = (self.imp_map.get_importance(offer) - self.MIN_IMPORTANCE) / (self.MAX_IMPORTANCE - self.MIN_IMPORTANCE)
            if ratio > self.reservationImportanceRatio + 0.2:
                return ResponseType.ACCEPT_OFFER
        self.lastReceivedBid = self.receivedBid
        return ResponseType.REJECT_OFFER

    def get_needed_random_bid(self, lower_ratio, upper_ratio):
        # print('called')
        p = np.arange(self.n_outcomes, 0, -1, dtype=float)
        p /= np.sum(p)
        lower_threshold = lower_ratio * (self.MAX_IMPORTANCE - self.MIN_IMPORTANCE) + self.MIN_IMPORTANCE
        upper_threshold = upper_ratio * (self.MAX_IMPORTANCE - self.MIN_IMPORTANCE) + self.MIN_IMPORTANCE

        for t in range(3):
            highest_opponent_importance = 0.0
            returned_bid = None
            # idx = np.random.choice(len(self.ami.outcomes), self.n_outcomes, replace=False)
            # random_outcomes = [self.ami.outcomes[_] for _ in idx]
            for bid in random.sample(self.ami.outcomes, self.n_outcomes):
                bid_importance = self.imp_map.get_importance(bid)
                bid_opponent_importance = self.opponent_imp_map.get_importance(bid)
                if lower_threshold <= bid_importance <= upper_threshold:
                    if self.offerRandomly:
                        bid = {i.name: bid[_] for _, i in enumerate(self.ami.issues)}
                        # print('bid',bid)
                        return bid
                    if bid_opponent_importance > highest_opponent_importance:
                        highest_opponent_importance = bid_opponent_importance
                        returned_bid = bid
            if returned_bid is not None:
                returned_bid = {i.name: returned_bid[_] for _, i in enumerate(self.ami.issues)}
                # print('returned_bid',returned_bid)
                return returned_bid

        # for t in range(3):
        #     k = 2 * self.n_outcomes
        #     highest_opponent_importance = 0.0
        #     returned_bid = None
        #     for i in range(k):
        #         bid = np.random.choice(self.ordered_outcomes_only, p=p)
        #         bid_importance = self.imp_map.get_importance(bid)
        #         bid_opponent_importance = self.opponent_imp_map.get_importance(bid)
        #         if lower_threshold <= bid_importance <= upper_threshold:
        #             if self.offerRandomly:
        #                 return bid
        #             if bid_opponent_importance > highest_opponent_importance:
        #                 highest_opponent_importance = bid_opponent_importance
        #                 returned_bid = bid
        #     if returned_bid is not None:
        #         return returned_bid
        # while(1):
        for bid in random.sample(self.ami.outcomes, self.n_outcomes):
            if self.imp_map.get_importance(bid) >= lower_threshold:
                bid = {i.name: bid[_] for _, i in enumerate(self.ami.issues)}
                # print('bid',bid)
                return bid
        print('None generated')

    def propose(self, state: MechanismState) -> Optional[Outcome]:
        if self.receivedBid is None:
            # print('1', self.MAX_IMPORTANCE_BID)
            return self.MAX_IMPORTANCE_BID
        bid = self.get_needed_random_bid(self.offer_lower_ratio, self.offer_higher_ratio)
        # print('2',bid)
        return bid


# class AgentGG(SAONegotiator):
#     def __init__(
#             self,
#             name: str = None,
#             parent: Controller = None,
#             ufun: Optional["UtilityFunction"] = None,
#             add_noise: bool = False
#     ):
#         self.add_noise = add_noise
#
#         self.ordered_outcomes = None
#         self.ordered_utils = None
#         self.n_outcomes = None
#         super().__init__(name=name, ufun=ufun, parent=parent)
#
#     def on_preferences_changed(self):
#         super().on_preferences_changed()
#         outcomes = self._nmi.discrete_outcomes()
#         self.ordered_outcomes = sorted(
#             [(self._preferences(outcome), outcome) for outcome in outcomes],
#             key=lambda x: x[0],
#             reverse=True,
#         )
#         self.ordered_utils = np.array([u for (u, _) in self.ordered_outcomes[::-1]])
#         self.n_outcomes = len(self.ordered_utils)
#
#     def respond(self, state: MechanismState, offer: "Outcome") -> "ResponseType":
#         if self._preferences is None:
#             return ResponseType.REJECT_OFFER
#         return ResponseType.REJECT_OFFER
#
#     def _get_bid_index(self, ulevel: float) -> int:
#         return max(0, self.n_outcomes - np.searchsorted(self.ordered_utils, ulevel) - 1)
#
#     def propose(self, state: MechanismState) -> Optional[Outcome]:
#         # ノイズsigma=0.005
#         if self.add_noise:
#             noise = np.random.normal(0.0, 0.005)
#         next_bid = self.ordered_outcomes[self._get_bid_index(self.target + noise)][1]
#         return next_bid
