from abc import ABCMeta, abstractmethod


class AbstractOpponentModel(metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, offer) -> float:
        raise NotImplementedError

    @abstractmethod
    def update(self, offer, t) -> None:
        raise NotImplementedError


# No knowledge about the preference profile
class NoModel(AbstractOpponentModel):
    def __call__(self, offer):
        pass

    def update(self, offer, t):
        pass


# Complex Automated Negotiations: Theories, Models, and Software Competitions
# Learns the issue weights based on how often the value of an issue changes
# The value weights are estimated based on the frequency they are offered
class HardHeadedFrequencyModel(AbstractOpponentModel):
    weights = {}
    evaluates = {}
    prevOffer = None

    def __init__(self, ufun, learn_coef=0.2, learn_value_addition=1):
        self.amountOfIssues = len(ufun.weights)
        self.learnCoef = learn_coef
        self.learnValueAddition = learn_value_addition
        self.gamma = 0.25
        self.goldenValue = self.learnCoef / self.amountOfIssues
        self.issues = ufun.issues
        for i in ufun.issues:
            self.weights[i.name] = 1.0 / self.amountOfIssues
            self.evaluates[i.name] = {v: 1.0 for v in i}

    def __call__(self, offer):
        util = 0
        for i, o in enumerate(offer):
            util += self.weights[self.issues[i].name] * (self.evaluates[self.issues[i].name][o] / max(self.evaluates[self.issues[i].name].values()))
        return util

    def update(self, offer, t):
        if self.prevOffer is not None:
            last_diff = self.determine_difference(offer, self.prevOffer)
            num_of_unchanged = len(last_diff) - sum(last_diff)
            total_sum = 1 + self.goldenValue * num_of_unchanged
            maximum_weight = 1 - self.amountOfIssues * self.goldenValue / total_sum
            for k, i in zip(self.weights.keys(), last_diff):
                weight = self.weights[k]
                if i == 0 and weight < maximum_weight:
                    self.weights[k] = (weight + self.goldenValue) / total_sum
                else:
                    self.weights[k] = weight / total_sum

        for issue, evaluator in offer.items():
            self.evaluates[issue][evaluator] += self.learnValueAddition
        self.prevOffer = offer

    @staticmethod
    def determine_difference(first, second):
        return [int(f == s) for f, s in zip(first.items(), second.items())]


# Counts how often each value is offered
# The utility of a bid is the sum of the score of its values divided by the best possible score
# The model only uses the first 100 unique bids for its estimation
class CUHKAgentValueModel(AbstractOpponentModel):
    evaluates = {}
    bid_history = []
    maximumBidsStored = 100
    maxPossibleTotal = 0

    def __init__(self, ufun):
        for i in ufun.issues:
            self.evaluates[i.name] = {v: 0.0 for v in i}

    def __call__(self, offer):
        total_bid_value = 0.0
        for issue in self.evaluates.keys():
            v = offer[issue]
            counter_per_value = self.evaluates[issue][v]
            total_bid_value += counter_per_value
        if total_bid_value == 0:
            return 0.0
        return total_bid_value / self.maxPossibleTotal

    def update(self, offer, t):
        if len(self.bid_history) > self.maximumBidsStored:
            return
        if offer not in self.bid_history:
            self.bid_history.append(offer)
        if len(self.bid_history) <= self.maximumBidsStored:
            self.update_statistics(offer)

    def update_statistics(self, offer):
        for issue in self.evaluates.keys():
            v = offer[issue]
            if self.evaluates[issue][v] + 1 > max(self.evaluates[issue].values()):
                self.maxPossibleTotal += 1
            self.evaluates[issue][v] += 1


# Defines the opponent’s utility as one minus the agent’s utility
class OppositeModel(AbstractOpponentModel):
    def __init__(self, my_ufun):
        self.ufun = my_ufun

    def __call__(self, offer):
        return 1. - self.ufun(offer)

    def update(self, offer, t):
        pass


# Perfect knowledge of the opponent’s preferences
class PerfectModel(AbstractOpponentModel):
    def __init__(self, opp_ufun):
        self.ufun = opp_ufun

    def __call__(self, offer):
        return self.ufun(offer)

    def update(self, offer, t):
        pass


# Defines the estimated utility as one minus the real utility
class WorstModel(AbstractOpponentModel):
    def __init__(self, opp_ufun):
        self.ufun = opp_ufun

    def __call__(self, offer):
        return 1. - self.ufun(offer)

    def update(self, offer, t):
        pass
