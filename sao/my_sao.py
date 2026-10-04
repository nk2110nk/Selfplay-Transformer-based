from __future__ import annotations
import time
from typing import List, Union, Tuple, Optional
from negmas.mechanisms import MechanismRoundResult
from negmas.sao import SAOMechanism, SAONegotiator, SAOResponse, ResponseType
import itertools
import pandas as pd
import math

import functools
import random
import time
from collections import defaultdict
from negmas.events import Event
from negmas.helpers import TimeoutCaller, TimeoutError, exception2str
from negmas.mechanisms import MechanismRoundResult
from negmas.outcomes.outcome_ops import cast_value_types, outcome_types_are_ok
from negmas.sao.common import ResponseType, SAOResponse, SAOState


class MySAOMechanism(SAOMechanism):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def __next__(self):
        result = self.step()
        if not self._current_state.running:
            raise StopIteration

        return result

    def reset(self):
        del self.nmi
        del self._current_proposer
        del self.agents_of_role
        del self._negotiators
        del self._history
        # del self._Mechanism__outcome_index
        # del self._Mechanism__outcomes
        # del self._state_factory

    def __call__(self, state: SAOState) -> MechanismRoundResult:
        """implements a round of the Stacked Alternating Offers Protocol."""
        state = self._current_state
        if self._frozen_neg_list is None:
            state.new_offers = []
        negotiators: list[SAONegotiator] = self.negotiators
        n_negotiators = len(negotiators)
        # times = dict(zip([_.id for _ in negotiators], itertools.repeat(0.0)))
        times = defaultdict(float, self._waiting_time)
        exceptions = dict(
            zip([_.id for _ in negotiators], [list() for _ in negotiators])
        )

        def _safe_counter(
            negotiator, *args, **kwargs
        ) -> tuple[SAOResponse | None, bool]:
            assert (
                not state.waiting or negotiator.id == state.current_proposer
            ), f"We are waiting with {state.current_proposer} as the last offerer but we are asking {negotiator.id} to offer\n{state}"
            rem = self.remaining_time
            if rem is None:
                rem = float("inf")
            timeout = min(
                self.nmi.negotiator_time_limit - times[negotiator.id],
                self.nmi.step_time_limit,
                rem,
                self._hidden_time_limit - self.time,
            )
            if timeout is None or timeout == float("inf") or self._sync_calls:
                __strt = time.perf_counter()
                try:
                    if (
                        negotiator == self._current_proposer
                    ) and self._offering_is_accepting:
                        self._current_state.n_acceptances = 0
                        response = negotiator(*args, **kwargs)
                    else:
                        response = negotiator(*args, **kwargs)
                except TimeoutError:
                    response = None
                    try:
                        negotiator.cancel()
                    except:
                        pass
                except Exception as ex:
                    exceptions[negotiator.id].append(exception2str())
                    if self.ignore_negotiator_exceptions:
                        self.announce(
                            Event(
                                "negotiator_exception",
                                {"negotiator": negotiator, "exception": ex},
                            )
                        )
                        times[negotiator.id] += time.perf_counter() - __strt
                        return SAOResponse(ResponseType.END_NEGOTIATION, None), True
                    else:
                        raise ex
                times[negotiator.id] += time.perf_counter() - __strt
            else:
                fun = functools.partial(negotiator, *args, **kwargs)
                __strt = time.perf_counter()
                try:
                    if (
                        negotiator == self._current_proposer
                    ) and self._offering_is_accepting:
                        state.n_acceptances = 0
                        response = TimeoutCaller.run(fun, timeout=timeout)
                    else:
                        response = TimeoutCaller.run(fun, timeout=timeout)
                except TimeoutError:
                    response = None
                except Exception as ex:
                    exceptions[negotiator.id].append(exception2str())
                    if self.ignore_negotiator_exceptions:
                        self.announce(
                            Event(
                                "negotiator_exception",
                                {"negotiator": negotiator, "exception": ex},
                            )
                        )
                        times[negotiator.id] += time.perf_counter() - __strt
                        return SAOResponse(ResponseType.END_NEGOTIATION, None), True
                    else:
                        raise ex
                times[negotiator.id] += time.perf_counter() - __strt
            if (
                self.check_offers
                and response is not None
                and response.outcome is not None
            ):
                if not self.outcome_space.is_valid(response.outcome):
                    return SAOResponse(response.response, None), False
                # todo: do not use .issues here as they are not guaranteed to exist (if it is not a cartesial outcome space)
                if self._enforce_issue_types and hasattr(self.outcome_space, "issues"):
                    if outcome_types_are_ok(
                        response.outcome, self.outcome_space.issues  # type: ignore
                    ):
                        return response, False
                    elif self._cast_offers:
                        return (
                            SAOResponse(
                                response.response,
                                cast_value_types(
                                    response.outcome, self.outcome_space.issues  # type: ignore
                                ),
                            ),
                            False,
                        )
                    return SAOResponse(response.response, None), False
            return response, False

        proposers, proposer_indices = [], []
        for i, neg in enumerate(negotiators):
            if not neg.capabilities.get("propose", False):
                continue
            proposers.append(neg)
            proposer_indices.append(i)
        n_proposers = len(proposers)
        if n_proposers < 1:
            if not self.dynamic_entry:
                state.broken = True
                state.has_error = True
                state.error_details = "No proposers and no dynamic entry"
                return MechanismRoundResult(state, times=times, exceptions=exceptions)
            else:
                return MechanismRoundResult(state, times=times, exceptions=exceptions)
        # if this is the first step (or no one has offered yet) which means that there is no _current_offer
        if (
            state.current_offer is None
            and n_proposers > 1
            and self._avoid_ultimatum
            and not self._ultimatum_avoided
        ):
            if not self.dynamic_entry and not self.state.step == 0:
                if self.end_negotiation_on_refusal_to_propose:
                    state.broken = True
                    return MechanismRoundResult(
                        state,
                        times=times,
                        exceptions=exceptions,
                    )
            # if we are trying to avoid an ultimatum, we take an offer from everyone and ignore them but one.
            # this way, the agent cannot know its order. For example, if we have two agents and 3 steps, this will
            # be the situation after each step:
            #
            # Case 1: Assume that it ignored the offer from agent 1
            # Step, Agent 0 calls received  , Agent 1 calls received    , relative time during last call
            # 0   , counter(None)->offer1*  , counter(None) -> offer0   , 0/3
            # 1   , counter(offer2)->offer3 , counter(offer1) -> offer2 , 1/3
            # 2   , counter(offer4)->offer5 , counter(offer3) -> offer4 , 2/3
            # 3   ,                         , counter(offer5)->offer6   , 3/3
            #
            # Case 2: Assume that it ignored the offer from agent 0
            # Step, Agent 0 calls received  , Agent 1 calls received    , relative time during last call
            # 0   , counter(None)->offer1   , counter(None) -> offer0*  , 0/3
            # 1   , counter(offer0)->offer2 , counter(offer2) -> offer3 , 1/3
            # 2   , counter(offer3)->offer4 , counter(offer4) -> offer5 , 2/3
            # 3   , counter(offer5)->offer6 ,                           , 3/3
            #
            # in both cases, the agent cannot know whether its last offer going to be passed to the other agent
            # (the ultimatum scenario) or not.
            responses = []
            responders = []
            for i, neg in enumerate(proposers):
                if not neg.capabilities.get("propose", False):
                    continue
                strt = time.perf_counter()
                resp, has_exceptions = _safe_counter(neg, state=self.state, offer=None)
                if has_exceptions:
                    state.broken = True
                    state.has_error = True
                    state.error_details = str(exceptions[neg.id])
                    return MechanismRoundResult(
                        state,
                        times=times,
                        exceptions=exceptions,
                    )
                if resp is None:
                    state.timedout = True
                    return MechanismRoundResult(
                        state,
                        times=times,
                        exceptions=exceptions,
                    )
                if resp.response != ResponseType.WAIT:
                    self._waiting_time[neg.id] = 0.0
                    self._waiting_start[neg.id] = float("inf")
                    self._frozen_neg_list = None
                else:
                    self._waiting_start[neg.id] = min(self._waiting_start[neg.id], strt)
                    self._waiting_time[neg.id] += (
                        time.perf_counter() - self._waiting_start[neg.id]
                    )
                if (
                    resp is None
                    or time.perf_counter() - strt > self.nmi.step_time_limit
                ):
                    state.timedout = True
                    return MechanismRoundResult(
                        state,
                        times=times,
                        exceptions=exceptions,
                    )
                if resp.response == ResponseType.END_NEGOTIATION:
                    state.broken = True
                    return MechanismRoundResult(
                        state,
                        times=times,
                        exceptions=exceptions,
                    )
                if resp.response in (ResponseType.NO_RESPONSE, ResponseType.WAIT):
                    continue
                if (
                    resp.response == ResponseType.REJECT_OFFER
                    and resp.outcome is None
                    and self.end_negotiation_on_refusal_to_propose
                ):
                    continue
                responses.append(resp)
                responders.append(i)
            if len(responses) < 1:
                if not self.dynamic_entry:
                    state.broken = True
                    state.has_error = True
                    state.error_details = "No proposers and no dynamic entry. This may happen if no negotiators responded to their first proposal request with an offer"
                    return MechanismRoundResult(
                        state,
                        times=times,
                        exceptions=exceptions,
                    )
                else:
                    return MechanismRoundResult(
                        state,
                        times=times,
                        exceptions=exceptions,
                    )
            # choose a random negotiator and set it as the current negotiator
            self._ultimatum_avoided = True
            selected = random.randint(0, len(responses) - 1)
            resp = responses[selected]
            neg = proposers[responders[selected]]
            _first_proposer = proposer_indices[responders[selected]]
            self._selected_first = _first_proposer
            self._last_checked_negotiator = _first_proposer
            state.current_offer = resp.outcome
            state.new_offers.append((neg.id, resp.outcome))
            self._current_proposer = neg
            state.current_proposer = neg.id
            state.n_acceptances = 1 if self._offering_is_accepting else 0
            if self._last_checked_negotiator >= 0:
                state.last_negotiator = self.negotiators[
                    self._last_checked_negotiator
                ].name
            else:
                state.last_negotiator = ""
            (
                self._current_proposer_agent,
                state.new_offerer_agents,
            ) = self._agent_info()

            # current_proposer_agent=current_proposer_agent,
            # new_offerer_agents=new_offerer_agents,
            return MechanismRoundResult(
                state,
                times=times,
                exceptions=exceptions,
            )

        # this is not the first round. A round will get n_negotiators responses
        neg_indx = (self._last_checked_negotiator + 1) % n_negotiators

        self._last_checked_negotiator = neg_indx
        neg = self.negotiators[neg_indx]
        strt = time.perf_counter()
        resp, has_exceptions = _safe_counter(
            neg, state=self.state, offer=state.current_offer
        )
        if has_exceptions:
            state.broken = True
            state.has_error = True
            state.error_details = str(exceptions[neg.id])
            return MechanismRoundResult(
                state,
                times=times,
                exceptions=exceptions,
            )
        if resp is None:
            state.timedout = True
            return MechanismRoundResult(
                state,
                times=times,
                exceptions=exceptions,
            )
        if resp.response == ResponseType.WAIT:
            self._waiting_start[neg.id] = min(self._waiting_start[neg.id], strt)
            self._waiting_time[neg.id] += time.perf_counter() - strt
            self._last_checked_negotiator = (neg_indx - 1) % n_negotiators
            offered = {self._negotiator_index[_[0]] for _ in state.new_offers}
            did_not_offer = sorted(
                list(set(range(n_negotiators)).difference(offered))
            )
            assert neg_indx in did_not_offer
            indx = did_not_offer.index(neg_indx)
            assert (
                self._frozen_neg_list is None
                or self._frozen_neg_list[0] == neg_indx
            )
            self._frozen_neg_list = did_not_offer[indx:] + did_not_offer[:indx]
            self._n_waits += 1
        else:
            self._stop_waiting(neg.id)

        if resp is None or time.perf_counter() - strt > self.nmi.step_time_limit:
            state.timedout = True
            return MechanismRoundResult(
                state,
                times=times,
                exceptions=exceptions,
            )
        if self._extra_callbacks:
            if state.current_offer is not None:
                for other in self.negotiators:
                    if other is not neg:
                        other.on_partner_response(
                            state=self.state,
                            partner_id=neg.id,
                            outcome=state.current_offer,
                            response=resp.response,
                        )
        if resp.response == ResponseType.NO_RESPONSE:
            pass
        if resp.response == ResponseType.WAIT:
            if self._n_waits > self._n_max_waits:
                self._stop_waiting(neg.id)
                state.timedout = True
                state.waiting = False
                return MechanismRoundResult(
                    state,
                    times=times,
                    exceptions=exceptions,
                )
            state.waiting = True
            return MechanismRoundResult(
                state,
                times=times,
                exceptions=exceptions,
            )
        if resp.response == ResponseType.END_NEGOTIATION:
            state.broken = True
            return MechanismRoundResult(
                state,
                times=times,
                exceptions=exceptions,
            )
        if resp.response == ResponseType.ACCEPT_OFFER:
            state.n_acceptances += 1
            if state.n_acceptances == n_negotiators:
                state.agreement = self._current_state.current_offer
                return MechanismRoundResult(
                    state,
                    timedout=False,
                    agreement=state.current_offer,
                    times=times,
                    exceptions=exceptions,
                    broken=False,
                )
        if resp.response == ResponseType.REJECT_OFFER:
            proposal = resp.outcome
            if (
                not self.allow_offering_just_rejected_outcome
                and proposal == state.current_offer
            ):
                proposal = None
            if proposal is None:
                if (
                    neg.capabilities.get("propose", True)
                    and self.end_negotiation_on_refusal_to_propose
                ):
                    state.broken = True
                    return MechanismRoundResult(
                        state,
                        times=times,
                        exceptions=exceptions,
                    )
                state.n_acceptances = 0
            else:
                state.n_acceptances = 1 if self._offering_is_accepting else 0
                if self._extra_callbacks:
                    for other in self.negotiators:
                        if other is neg:
                            continue
                        other.on_partner_proposal(
                            partner_id=neg.id, offer=proposal, state=self.state
                        )
            state.current_offer = proposal
            self._current_proposer = neg
            state.current_proposer = neg.id
            state.new_offers.append((neg.id, proposal))
            if self._last_checked_negotiator >= 0:
                state.last_negotiator = self.negotiators[
                    self._last_checked_negotiator
                ].name
            else:
                state.last_negotiator = ""
            (
                self._current_proposer_agent,
                state.new_offerer_agents,
            ) = self._agent_info()

        return MechanismRoundResult(
            state,
            times=times,
            exceptions=exceptions,
        )
    
    def step(self):
        """Runs a single step of the mechanism.

        Returns:
            MechanismState: The state of the negotiation *after* the round is conducted

        Remarks:

            - Every call yields the results of one round (see `round()`)
            - If the mechanism was yet to start, it will start it and runs one round
            - There is another function (`run()`) that runs the whole mechanism in blocking mode
        """
        if self._start_time is None or self._start_time < 0:
            self._start_time = time.perf_counter()
        self.checkpoint_on_step_started()
        state = self.state
        state4history = self.state4history

        # end with a timeout if condition is met
        if (
            (self.time > self.time_limit)
            or (self.nmi.n_steps and self._current_state.step >= self.nmi.n_steps)
            or self.time > self._hidden_time_limit
        ):
            (
                self._current_state.running,
                self._current_state.broken,
                self._current_state.timedout,
            ) = (False, False, True)
            self.on_negotiation_end()
            return self.state

        # if there is a single negotiator and no other negotiators can be added,
        # end without starting
        if len(self._negotiators) < 2:
            if self.nmi.dynamic_entry:
                return self.state
            else:
                (
                    self._current_state.running,
                    self._current_state.broken,
                    self._current_state.timedout,
                ) = (False, False, False)
                self.on_negotiation_end()
                return self.state

        # if the mechanism states that it is broken, timedout or ended with
        # agreement, report that
        if (
            self._current_state.broken
            or self._current_state.timedout
            or self._current_state.agreement is not None
        ):
            self._current_state.running = False
            self.on_negotiation_end()
            return self.state

        if not self._current_state.running:
            # if we did not start, just start
            self._current_state.running = True
            self._current_state.step = 0
            self._start_time = time.perf_counter()
            self._current_state.started = True
            state = self.state
            # if the mechanism indicates that it cannot start, keep trying
            if self.on_negotiation_start() is False:
                (
                    self._current_state.agreement,
                    self._current_state.broken,
                    self._current_state.timedout,
                ) = (None, False, False)
                return self.state
            for a in self.negotiators:
                a._on_negotiation_start(state=state)
            self.announce(Event(type="negotiation_start", data=None))
        else:
            # if no steps are remaining, end with a timeout
            remaining_steps, remaining_time = self.remaining_steps, self.remaining_time
            if (remaining_steps is not None and remaining_steps <= 0) or (
                remaining_time is not None and remaining_time <= 0.0
            ):
                self._current_state.running = False
                (
                    self._current_state.agreement,
                    self._current_state.broken,
                    self._current_state.timedout,
                ) = (None, False, True)
                self.on_negotiation_end()
                return self.state

        # send round start only if the mechanism is not waiting for anyone
        # TODO check this.
        if not self._current_state.waiting and self._extra_callbacks:
            for agent in self._negotiators:
                agent.on_round_start(state)

        # run a round of the mechanism and get the new state
        step_start = (
            time.perf_counter() if not self._current_state.waiting else self._last_start
        )
        self._last_start = step_start
        self._current_state.waiting = False
        result = self(self._current_state)
        self._current_state = result.state
        step_time = time.perf_counter() - step_start
        self._stats["round_times"].append(step_time)

        # if negotaitor times are reported, save them
        if result.times:
            for k, v in result.times.items():
                if v is not None:
                    self._stats["times"][k] += v
        # if negotaitor exceptions are reported, save them
        if result.exceptions:
            for k, v in result.exceptions.items():
                if v:
                    self._stats["exceptions"][k] += v

        # update current state variables from the result of the round just run
        (
            self._current_state.has_error,
            self._current_state.error_details,
            self._current_state.waiting,
        ) = (
            result.state.has_error,
            result.state.error_details,
            result.state.waiting,
        )
        if self._current_state.has_error:
            self.on_mechanism_error()
        if (
            self.nmi.step_time_limit is not None
            and step_time > self.nmi.step_time_limit
        ):
            (
                self._current_state.broken,
                self._current_state.timedout,
                self._current_state.agreement,
            ) = (False, True, None)
        else:
            (
                self._current_state.broken,
                self._current_state.timedout,
                self._current_state.agreement,
            ) = (
                result.state.broken,
                result.state.timedout,
                result.state.agreement,
            )
        if (
            (self._current_state.agreement is not None)
            or self._current_state.broken
            or self._current_state.timedout
        ):
            self._current_state.running = False

        # now switch to the new state
        state = self.state
        if not self._current_state.waiting and result.completed:
            state4history = self.state4history
            if self._extra_callbacks:
                for agent in self._negotiators:
                    agent.on_round_end(state)
            self._add_to_history(state4history)
            # we only indicate a new step if no one is waiting
            self._current_state.step += 1 # 変更箇所
            self._current_state.time = self.time
            self._current_state.relative_time = self.relative_time

        if not self._current_state.running:
            self.on_negotiation_end()
        return self.state

"""
    def plot(
        self,
        visible_negotiators: Union[Tuple[int, int], Tuple[str, str]] = (0, 1),
        plot_utils=True,
        plot_outcomes=False,
        utility_range: Optional[Tuple[float, float]] = None,
        path: Optional[str] = None
    ):
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        import matplotlib.ticker as tick

        if self.issues is not None and len(self.issues) > 1:
            plot_outcomes = False

        if len(self.negotiators) < 2:
            print("Cannot visualize negotiations with more less than 2 negotiators")
            return
        if len(visible_negotiators) > 2:
            print("Cannot visualize more than 2 agents")
            return
        if isinstance(visible_negotiators[0], str):
            tmp = []
            for _ in visible_negotiators:
                for n in self.negotiators:
                    if n.id == _:
                        tmp.append(n)
        else:
            visible_negotiators = [
                self.negotiators[visible_negotiators[0]],
                self.negotiators[visible_negotiators[1]],
            ]
        indx = dict(zip([_.id for _ in self.negotiators], range(len(self.negotiators))))
        history = []
        for state in self.history:
            for a, o in state.new_offers:
                o = tuple(v for k, v in o.items())
                history.append(
                    {
                        "current_proposer": a,
                        "current_offer": o,
                        "offer_index": self.outcomes.index(o),
                        "relative_time": state.relative_time,
                        "step": state.step,
                        "u0": visible_negotiators[0].ufun(o),
                        "u1": visible_negotiators[1].ufun(o),
                        "agreement": state.agreement,
                    }
                )
        history = pd.DataFrame(data=history)
        has_history = len(history) > 0
        has_front = 1
        n_negotiators = len(self.negotiators)
        n_agents = len(visible_negotiators)
        ufuns = self._get_preferences()
        outcomes = self.outcomes
        utils = [tuple(f(o) for f in ufuns) for o in outcomes]
        agent_names = [a.name for a in visible_negotiators]
        if has_history:
            history["offer_index"] = [outcomes.index(_) for _ in history.current_offer]
        frontier, frontier_outcome = self.pareto_frontier(sort_by_welfare=True)
        frontier_indices = [
            i
            for i, _ in enumerate(frontier)
            if _[0] is not None
            and _[0] > float("-inf")
            and _[1] is not None
            and _[1] > float("-inf")
        ]
        frontier = [frontier[i] for i in frontier_indices]
        max_welfare = frontier[0]
        frontier = sorted(frontier, key=lambda x: x[0])
        frontier_outcome = [frontier_outcome[i] for i in frontier_indices]
        frontier_outcome_indices = [outcomes.index(_) for _ in frontier_outcome]
        if plot_utils:
            fig_util = plt.figure(figsize=(8.0, 4.8))
        if plot_outcomes:
            fig_outcome = plt.figure(figsize=(8.0, 4.8))
        gs_util = gridspec.GridSpec(n_agents, has_front * 3 + 2) if plot_utils else None
        gs_outcome = (
            gridspec.GridSpec(n_agents, has_front * 3 + 2) if plot_outcomes else None
        )
        axs_util, axs_outcome = [], []

        for a in range(n_agents):
            if a == 0:
                if plot_utils:
                    axs_util.append(fig_util.add_subplot(gs_util[a, -2:]))
                if plot_outcomes:
                    axs_outcome.append(
                        fig_outcome.add_subplot(gs_outcome[a, -2:])
                    )
            else:
                if plot_utils:
                    axs_util.append(
                        fig_util.add_subplot(gs_util[a, -2:], sharex=axs_util[0])
                    )
                if plot_outcomes:
                    axs_outcome.append(
                        fig_outcome.add_subplot(
                            gs_outcome[a, -2:], sharex=axs_outcome[0]
                        )
                    )
            if plot_utils:
                clrs = ("blue", "green")
                axs_util[-1].set_ylabel(agent_names[a] + "\'s Utility", color=clrs[a])
                # if a != 0:
                #     axs_util[-1].set_xlabel('Time')
                # if a == 0:
                #     axs_util[-1].set_title("Time-Util Graph")
            if plot_outcomes:
                axs_outcome[-1].set_ylabel(agent_names[a] + "\'s Utility")
        for a, (au, ao) in enumerate(
            zip(
                itertools.chain(axs_util, itertools.repeat(None)),
                itertools.chain(axs_outcome, itertools.repeat(None)),
            )
        ):
            if au is None and ao is None:
                break
            if has_history:
                h = history.loc[
                    history.current_proposer == visible_negotiators[a].id,
                    ["relative_time", "offer_index", "current_offer"],
                ]
                h["utility"] = h["current_offer"].apply(ufuns[a])
                h_opp = history.loc[
                    history.current_proposer != visible_negotiators[a].id,
                    ["relative_time", "offer_index", "current_offer"],
                ]
                h_opp["utility"] = h_opp["current_offer"].apply(ufuns[a])
                if plot_outcomes:
                    ao.plot(h.relative_time, h["offer_index"], marker=',')
                if plot_utils:
                    if a == 0:
                        clrs = ("blue", "green")
                    else:
                        clrs = ("green", "blue")
                    au.plot(h.relative_time, h.utility, marker=',', color=clrs[0])
                    au.plot(h_opp.relative_time, h_opp.utility, marker=',', color=clrs[1])
                    au.set_xlabel('Relative Time')
                    if utility_range is not None:
                        au.set_ylim(*utility_range)
                    else:
                        au.set_ylim(0.0, 1.05)

        if has_front:
            if plot_utils:
                axu = fig_util.add_subplot(gs_util[:, 0:-2])
                axu.scatter(
                    [_[0] for _ in utils],
                    [_[1] for _ in utils],
                    label="Outcomes",
                    color="pink",
                    marker=".",
                    s=10,
                    zorder=0
                )
            if plot_outcomes:
                axo = fig_outcome.add_subplot(gs_outcome[:, 0:-2])
            clrs = ("blue", "green")
            mkr = ('v', '^')
            if plot_utils:
                f1, f2 = [_[0] for _ in frontier], [_[1] for _ in frontier]
                # axu.scatter(f1, f2, label="frontier", color="red", marker="x")
                axu.plot(f1, f2, linewidth=1.0, label="Pareto", color="magenta", marker="o", markersize=6, zorder=1)
                # axu.legend(loc='lower left')
                axu.set_xlabel(agent_names[0] + "\'s Utility")
                axu.xaxis.label.set_color(clrs[0])
                axu.set_ylabel(agent_names[1] + "\'s Utility")
                axu.yaxis.label.set_color(clrs[1])
                # axu.set_title("Outcome Space")
                if self.agreement is not None:
                    pareto_distance = 1e9
                    agreement_tuple = tuple(v for k, v in self.agreement.items())
                    cu = (ufuns[0](agreement_tuple), ufuns[1](agreement_tuple))
                    for pu in frontier:
                        dist = math.sqrt((pu[0] - cu[0]) ** 2 + (pu[1] - cu[1]) ** 2)
                        if dist < pareto_distance:
                            pareto_distance = dist
                    # axu.text(
                    #     0.03,
                    #     0.18,
                    #     f"Pareto-distance={pareto_distance:5.2}",
                    #     verticalalignment="top",
                    #     transform=axu.transAxes,
                    # )

            if plot_outcomes:
                axo.scatter(
                    frontier_outcome_indices,
                    frontier_outcome_indices,
                    color="magenta",
                    marker="o",
                    label="Pareto",
                    s=6,
                    zorder=1
                )
                axo.legend(loc='lower left')
                axo.set_xlabel(agent_names[0])
                axo.set_ylabel(agent_names[1])

            if plot_utils and has_history:
                axu.scatter(
                    [max_welfare[0]],
                    [max_welfare[1]],
                    color="black",
                    marker="D",
                    label=f"Nash",
                    zorder=2
                )

                for a in range(n_agents):
                    h = history.loc[
                        (history.current_proposer == self.negotiators[a].id)
                        | ~(history["agreement"].isnull()),
                        ["relative_time", "offer_index", "current_offer"],
                    ]
                    h["u0"] = h["current_offer"].apply(ufuns[0])
                    h["u1"] = h["current_offer"].apply(ufuns[1])

                    # make color map
                    import matplotlib.collections as mcoll
                    import matplotlib.path as mpath
                    from matplotlib.colors import LinearSegmentedColormap as lsc
                    import numpy as np

                    def colorline(x, y, cmap=plt.get_cmap('copper'), norm=plt.Normalize(0.0, 1.0)):
                        pth = mpath.Path(np.column_stack([h.u0, h.u1]))
                        verts = pth.interpolated(steps=3).vertices
                        x, y = verts[:, 0], verts[:, 1]
                        z = np.linspace(0, 1, len(x))
                        segments = make_segments(x, y)
                        lc = mcoll.LineCollection(
                            segments,
                            array=z,
                            cmap=cmap,
                            norm=norm,
                            linewidth=1,
                            zorder=3,
                        )
                        axu.add_collection(lc)
                        return lc

                    def make_segments(x, y):
                        if len(x) == 1:
                            return np.array([[(u0, u1) for u0, u1 in zip(x, y)]])
                        points = np.array([x, y]).T.reshape(-1, 1, 2)
                        segments = np.concatenate([points[:-1], points[1:]], axis=1)
                        return segments

                    cmap = lsc.from_list('colormap_name', ['light' + clrs[a], clrs[a]])
                    if len(h.u0) != 0:
                        lc = mcoll.LineCollection(
                            make_segments(h.u0, h.u1),
                            array=np.linspace(0, 1, len(h.u0)),
                            cmap=cmap,
                            norm=plt.Normalize(0.0, 1.0),
                            linewidth=1,
                            zorder=3,
                        )
                        axu.add_collection(lc)
                        # colorline(h.u0, h.u1, cmap=cmap)

                    axu.scatter(
                        h.u0,
                        h.u1,
                        c=np.linspace(0, 1, len(h.u0)),
                        marker=mkr[a],
                        cmap=cmap,
                        # label=f"{agent_names[a]}",
                        s=5**2,
                        zorder=3
                    )

                    # axu.plot(
                    #     h.u0,
                    #     h.u1,
                    #     linewidth=1,
                    #     marker=mkr[a],
                    #     color=clrs[a],
                    #     label=f"{agent_names[a]}",
                    #     markersize=5,
                    #     zorder=3
                    # )
            if plot_outcomes and has_history:
                steps = sorted(history.step.unique().tolist())
                aoffers = [[], []]
                for step in steps[::2]:
                    offrs = []
                    for a in range(n_agents):
                        a_offer = history.loc[
                            (history.current_proposer == agent_names[a])
                            & ((history.step == step) | (history.step == step + 1)),
                            "offer_index",
                        ]
                        if len(a_offer) > 0:
                            offrs.append(a_offer.values[-1])
                    if len(offrs) == 2:
                        aoffers[0].append(offrs[0])
                        aoffers[1].append(offrs[1])
                axo.scatter(aoffers[0], aoffers[1], color=clrs[0], label=f"offers")

            if self.state.agreement is not None:
                agreement_tuple = tuple(v for k, v in self.state.agreement.items())
                if plot_utils:
                    axu.scatter(
                        [ufuns[0](agreement_tuple)],
                        [ufuns[1](agreement_tuple)],
                        color="red",
                        marker="s",
                        s=50,
                        label="Agreement",
                        zorder=4
                    )
                    axu.legend(loc='lower left')
                if plot_outcomes:
                    axo.scatter(
                        [outcomes.index(self.state.agreement)],
                        [outcomes.index(self.state.agreement)],
                        color="red",
                        marker="s",
                        s=50,
                        label="Agreement",
                        zorder=4
                    )

        if plot_utils:
            for ax in fig_util.get_axes():
                ax.xaxis.set_minor_locator(tick.MultipleLocator(0.05))
                ax.yaxis.set_minor_locator(tick.MultipleLocator(0.05))
                ax.set_xlim(0, 1.05)
                ax.set_ylim(0, 1.05)
                ax.grid(color='gray', which='both', alpha=0.1, linestyle='--')
            fig_util.tight_layout()
            if path is not None:
                fig_util.savefig(path)
            else:
                fig_util.show()
        if plot_outcomes:
            for ax in fig_outcome.get_axes():
                ax.xaxis.set_minor_locator(tick.MultipleLocator(0.05))
                ax.yaxis.set_minor_locator(tick.MultipleLocator(0.05))
                ax.set_xlim(0, 1.05)
                ax.set_ylim(0, 1.05)
                ax.grid(color='gray', which='both', alpha=0.1, linestyle='--')
            fig_outcome.tight_layout()
            if path is not None:
                fig_outcome.savefig(path)
            else:
                fig_outcome.show()
"""