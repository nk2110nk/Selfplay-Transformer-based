import itertools
import xml.etree.ElementTree as ET
from functools import reduce
from operator import mul
from typing import Optional, Dict, List
from copy import deepcopy

import numpy as np

from negmas.generics import ienumerate, ivalues, ikeys
from negmas.outcomes import Issue, outcome_is_valid, Outcome
from negmas.utilities import (
    HyperRectangleUtilityFunction,
    LinearUtilityAggregationFunction,
    MappingUtilityFunction,
    UtilityValue,
)
from negmas.preferences.complex import WeightedUtilityFunction


@classmethod
def from_xml_str(
        cls,
        xml_str: str,
        domain_issues: Optional[List[Issue]] = None,
        force_single_issue=False,
        force_numeric=False,
        keep_issue_names=True,
        keep_value_names=True,
        safe_parsing=True,
        normalize_utility=True,
        geniusize_utility=True,
        max_n_outcomes: int = 1e6,
        ignore_discount=False,
        ignore_reserved=False,
        name=None,
):
    domain_issues = list(domain_issues)
    print(domain_issues)
    issue_list = deepcopy(domain_issues)
    root = ET.fromstring(xml_str)
    if safe_parsing and root.tag != "utility_space":
        raise ValueError(f"Root tag is {root.tag}: Expected utility_space")

    if domain_issues is not None:
        if isinstance(domain_issues, list):
            domain_issues: Dict[str, Issue] = dict(
                zip([_.name for _ in domain_issues], domain_issues)
            )
        elif isinstance(domain_issues, Issue) and force_single_issue:
            domain_issues = dict(zip([domain_issues.name], [domain_issues]))
    objective = None
    reserved_value = 0.0
    discount_factor = 0.0
    for child in root:
        if child.tag == "objective":
            objective = child
        elif child.tag == "reservation":
            reserved_value = float(child.attrib["value"])
        elif child.tag == "discount_factor":
            discount_factor = float(child.attrib["value"])

    if objective is None:
        if safe_parsing:
            pass
            # raise ValueError(f'No objective child was found in the root')
        objective = root
    weights = {}
    issues = {}
    real_issues = {}
    issue_info = {}
    issue_keys = {}
    rects, rect_utils = [], []

    def _get_hyperrects(ufun, max_utility, utiltype=float):
        utype = ufun.attrib.get("type", "none")
        uweight = float(ufun.attrib.get("weight", 1))
        uagg = ufun.attrib.get("aggregation", "sum")
        if uagg != "sum":
            raise ValueError(
                f"Hypervolumes combined using {uagg} are not supported (only sum is supported)"
            )
        total_util = utiltype(0)
        rects = []
        rect_utils = []
        if utype == "PlainUfun":
            for rect in ufun:
                util = utiltype(rect.attrib.get("utility", 0))
                total_util += util if util > 0 else 0
                ranges = {}
                rect_utils.append(util * uweight)
                for r in rect:
                    key = issue_keys[int(r.attrib["index"]) - 1]
                    ranges[key] = (
                        utiltype(r.attrib["min"]),
                        utiltype(r.attrib["max"]),
                    )
                rects.append(ranges)
        else:
            raise ValueError(f"Unknown ufun type {utype}")
        total_util = total_util if max_utility is None else max_utility
        if normalize_utility:
            for i, u in enumerate(rect_utils):
                rect_utils[i] = u / total_util
        return rects, rect_utils

    for child in objective:
        if child.tag == "weight":
            indx = int(child.attrib["index"]) - 1
            weights[indx] = float(child.attrib["value"])
        elif child.tag == "utility_function" or child.tag == "utility":
            utility_tag = child
            max_utility = child.attrib.get("maxutility", None)
            if max_utility is not None:
                max_utility = float(max_utility)
            ufun_found = False
            for ufun in utility_tag:
                if ufun.tag == "ufun":
                    ufun_found = True
                    _r, _u = _get_hyperrects(ufun, max_utility)
                    rects += _r
                    rect_utils += _u
            if not ufun_found:
                raise ValueError(
                    f"Cannot find ufun tag inside a utility_function tag"
                )

        elif child.tag == "issue":
            indx = int(child.attrib["index"]) - 1
            myname = child.attrib["name"]
            issue_key = myname if keep_issue_names else indx
            if domain_issues is not None and myname not in domain_issues.keys():
                raise ValueError(
                    f"Issue {myname} is not in the input issue names ({domain_issues.keys()})"
                )
            issue_info[issue_key] = {"name": myname, "index": indx}
            issue_keys[indx] = issue_key
            info = {"type": "discrete", "etype": "discrete", "vtype": "discrete"}
            for a in ("type", "etype", "vtype"):
                info[a] = child.attrib.get(a, info[a])
            mytype = info["type"]
            value_scale = None
            value_shift = None
            if mytype == "discrete":
                issues[issue_key] = {}
                if (
                        domain_issues is not None
                        and domain_issues[myname].is_continuous()
                ):
                    raise ValueError(
                        f"Got a {mytype} issue but expected a continuous valued issue"
                    )
                # issues[indx]['items'] = {}
            elif mytype in ("integer", "real"):
                lower, upper = (
                    child.attrib.get("lowerbound", None),
                    child.attrib.get("upperbound", None),
                )
                for rng_child in child:
                    if rng_child.tag == "range":
                        lower, upper = (
                            rng_child.attrib.get("lowerbound", lower),
                            rng_child.attrib.get("upperbound", upper),
                        )
                if mytype == "integer":
                    issues[issue_key] = {}
                    if (
                            domain_issues is not None
                            and domain_issues[myname].is_continuous()
                    ):
                        raise ValueError(
                            f"Got a {mytype} issue but expected a continuous valued issue"
                        )
                    # issues[indx]['items'] = {}
                    lower, upper = int(lower), int(upper)
                    for i in range(lower, upper + 1):
                        if domain_issues is not None and not outcome_is_valid(
                                (i,), [domain_issues[myname]]
                        ):
                            raise ValueError(
                                f"Value {i} is not in the domain issue values: "
                                f"{domain_issues[myname].values}"
                            )
                        issues[issue_key][i] = i if keep_value_names else i - lower
                else:
                    lower, upper = float(lower), float(upper)
                    if (
                            domain_issues is not None
                            and not domain_issues[myname].is_continuous()
                    ):
                        n_steps = domain_issues[myname].cardinality()
                        delta = (n_steps - 1) / (upper - lower)
                        value_shift = -lower * delta
                        value_scale = delta
                        lower, upper = 0, n_steps - 1
                        issues[issue_key] = {}
                        for i in range(lower, upper + 1):
                            issues[issue_key][i] = (
                                str(i) if keep_value_names else i - lower
                            )
                    else:
                        real_issues[issue_key] = {}
                        real_issues[issue_key]["range"] = (lower, upper)
                        real_issues[issue_key]["key"] = issue_key
            else:
                raise ValueError(f"Unknown type: {mytype}")
            if mytype in "discrete" or "integer" or "real":
                found_values = False
                for item in child:
                    if item.tag == "item":
                        if mytype == "real":
                            raise ValueError(
                                f"cannot specify item utilities for real type"
                            )
                        item_indx = int(item.attrib["index"]) - 1
                        item_name: str = item.attrib.get("value", None)
                        if item_name is None:
                            continue
                        item_key = (
                            item_name
                            if keep_value_names
                               and item_name is not None
                               and not force_numeric
                            else item_indx
                        )
                        if domain_issues is not None:
                            domain_all = list(domain_issues[myname].all)
                            if len(domain_all) > 0 and isinstance(
                                    domain_all[0], int
                            ):
                                item_key = int(item_key)
                            if len(domain_all) > 0 and isinstance(
                                    domain_all[0], int
                            ):
                                item_name = int(item_name)
                            if item_name not in domain_all:
                                raise ValueError(
                                    f"Value {item_name} is not in the domain issue values: "
                                    f"{domain_issues[myname].values}"
                                )
                            if len(domain_all) > 0 and isinstance(
                                    domain_all[0], int
                            ):
                                item_name = str(item_name)
                        if mytype == "integer":
                            item_key = int(item_key)
                        issues[issue_key][item_key] = float(
                            item.attrib.get("evaluation", reserved_value)
                        )
                        found_values = True
                    elif item.tag == "evaluator":
                        if item.attrib["ftype"] == "linear":
                            offset = item.attrib.get(
                                "offset", item.attrib.get("parameter0", 0.0)
                            )
                            slope = item.attrib.get(
                                "slope", item.attrib.get("parameter1", 1.0)
                            )
                            offset, slope = float(offset), float(slope)
                            if value_scale is None:
                                fun = lambda x: offset + slope * float(x)
                            else:
                                fun = lambda x: offset + slope * (
                                        value_scale * float(x) + value_shift
                                )
                        elif item.attrib["ftype"] == "triangular":
                            strt = item.attrib.get("parameter0", 0.0)
                            end = item.attrib.get("parameter1", 1.0)
                            middle = item.attrib.get("parameter2", 1.0)
                            strt, end, middle = (
                                float(strt),
                                float(end),
                                float(middle),
                            )
                            offset1, slope1 = strt, (middle - strt)
                            offset2, slope2 = middle, (middle - end)
                            if value_scale is None:
                                fun = (
                                    lambda x: offset1 + slope1 * float(x)
                                    if x < middle
                                    else offset2 + slope2 * float(x)
                                )
                            else:
                                fun = (
                                    lambda x: offset1
                                              + slope1
                                              * (value_scale * float(x) + value_shift)
                                    if x < middle
                                    else offset2
                                         + slope2
                                         * (value_scale * float(x) + value_shift)
                                )
                        else:
                            raise ValueError(
                                f'Unknown ftype {item.attrib["ftype"]}'
                            )
                        if mytype == "real" and value_scale is None:
                            real_issues[issue_key]["fun"] = fun
                        else:
                            for item_key, value in issues[issue_key].items():
                                issues[issue_key][item_key] = fun(value)
                            found_values = True
                if not found_values and issue_key in issues.keys():
                    issues.pop(issue_key, None)
            else:
                """Here goes the code for real-valued issues"""

    if geniusize_utility and normalize_utility:
        for key, issue in ienumerate(issues):
            factor = max(issues[key].values())
            for item_key in ikeys(issue):
                issues[key][item_key] = (
                        issues[key][item_key] / factor
                )

    # if not keep_issue_names:
    #    issues = [issues[_] for _ in issues.keys()]
    #    real_issues = [real_issues[_] for _ in sorted(real_issues.keys())]
    #    for i, issue in enumerate(issues):
    #        issues[i] = [issue[_] for _ in issue.keys()]

    if safe_parsing and (
            len(weights) > 0
            and len(weights) != len(issues) + len(real_issues)
            and len(weights) != len(issues)
    ):
        raise ValueError(
            f"Got {len(weights)} weights for {len(issues)} issues and {len(real_issues)} real issues"
        )

    if force_single_issue and (
            len(rects) > 0
            or len(real_issues) > 1
            or (len(real_issues) > 0 and len(issues) > 0)
    ):
        raise ValueError(
            f"Cannot force single issue with a hyper-volumes based function"
        )

    # add utilities specified not as hyper-rectangles
    u = None
    if len(issues) > 0:
        if len(weights) > 0:
            for key, issue in zip(ikeys(issues), ivalues(issues)):
                try:
                    w = weights[issue_info[key]["index"]]
                except:
                    w = 1.0
                for item_key in ikeys(issue):
                    issue[item_key] *= w
        if force_single_issue:
            n_outcomes = None
            if max_n_outcomes is not None:
                n_items = [len(_) for _ in ivalues(issues)]
                n_outcomes = reduce(mul, n_items, 1)
                if n_outcomes > max_n_outcomes:
                    return None, reserved_value, discount_factor
            if keep_value_names:
                names = itertools.product(
                    *[
                        [
                            str(item_key).replace("&", "-")
                            for item_key in ikeys(items)
                        ]
                        for issue_key, items in zip(ikeys(issues), ivalues(issues))
                    ]
                )
                names = map(lambda items: ("+".join(items),), names)
            else:
                if n_outcomes is None:
                    n_items = [len(_) for _ in ivalues(issues)]
                    n_outcomes = reduce(mul, n_items, 1)
                names = [(_,) for _ in range(n_outcomes)]
            utils = itertools.product(
                *[
                    [item_utility for item_utility in ivalues(items)]
                    for issue_key, items in zip(ikeys(issues), ivalues(issues))
                ]
            )
            utils = map(lambda vals: sum(vals), utils)
            if normalize_utility:
                utils = list(utils)
                umax, umin = max(utils), min(utils)
                if umax != umin:
                    utils = [(_ - umin) / (umax - umin) for _ in utils]
            if keep_issue_names:
                u = MappingUtilityFunction(dict(zip(names, utils)))
            else:
                u = MappingUtilityFunction(dict(zip(names, utils)))
        else:
            utils = None
            if normalize_utility:
                utils = itertools.product(
                    *[
                        [item_utility for item_utility in ivalues(items)]
                        for issue_key, items in zip(ikeys(issues), ivalues(issues))
                    ]
                )
                utils = list(map(lambda vals: sum(vals), utils))
                if not geniusize_utility:
                    umax, umin = max(utils), min(utils)
                    factor = umax - umin
                    if factor > 1e-8:
                        offset = (umin / len(issues)) / factor
                    else:
                        offset = 0.0
                        factor = 1.0
                    for key, issue in ienumerate(issues):
                        for item_key in ikeys(issue):
                            issues[key][item_key] = (
                                    issues[key][item_key] / factor - offset
                            )
            if len(issues) > 1:
                print(issues)
                u = LinearUtilityAggregationFunction(values=issues, issues=issue_list)
            else:
                first_key = list(ikeys(issues))[0]
                if utils is None:
                    utils = ivalues(issues[first_key])
                if keep_issue_names:
                    u = MappingUtilityFunction(
                        dict(zip([(_,) for _ in ikeys(issues[first_key])], utils))
                    )
                else:
                    u = MappingUtilityFunction(
                        dict(zip([(_,) for _ in range(len(utils))], utils))
                    )

    # add real_valued issues
    if len(real_issues) > 0:
        if len(weights) > 0:
            for key, issue in zip(ikeys(real_issues), ivalues(real_issues)):
                try:
                    w = weights[issue_info[key]["index"]]
                except:
                    w = 1.0
                issue["fun_final"] = lambda x: w * issue["fun"](x)
        if normalize_utility:
            n_items_to_test = 10
            utils = itertools.product(
                *[
                    [
                        issue["fun"](_)
                        for _ in np.linspace(
                        issue["range"][0],
                        issue["range"][1],
                        num=n_items_to_test,
                        endpoint=True,
                    )
                    ]
                    for key, issue in zip(ikeys(real_issues), ivalues(real_issues))
                ]
            )
            utils = list(map(lambda vals: sum(vals), utils))
            umax, umin = max(utils), min(utils)
            factor = umax - umin
            if factor > 1e-8:
                offset = (umin / len(real_issues)) / factor
            else:
                offset = 0.0
                factor = 1.0
            for key, issue in real_issues.items():
                issue["fun_final"] = lambda x: w * issue["fun"](x) / factor - offset
        u_real = LinearUtilityAggregationFunction(
            values={_["key"]: _["fun_final"] for _ in real_issues.values()}, issues=issue_list 
        )
        if u is None:
            u = u_real
        else:
            u = WeightedUtilityFunction(
                ufuns=[u, u_real], weights=[1.0, 1.0]
            )

    # add hyper rectangles issues
    if len(rects) > 0:
        uhyper = HyperRectangleUtilityFunction(
            outcome_ranges=rects, utilities=rect_utils
        )
        if u is None:
            u = uhyper
        else:
            u = WeightedUtilityFunction(
                ufuns=[u, uhyper], weights=[1.0, 1.0]
            )
    if reserved_value is not None and not ignore_reserved and u is not None:
        u.reserved_value = reserved_value
    if ignore_discount:
        discount_factor = None
    return u, discount_factor


def luaf_call(self, offer: Optional["Outcome"]) -> Optional[UtilityValue]:
    if offer is None:
        return self.reserved_value
    # u = ExactUtilityValue(0.0)
    u = 0.0
    for k in self.issue_utilities.keys():
        u += self.weights[k] * self.issue_utilities[k](offer[k])
    return u


def muf_call(self, offer: Optional["Outcome"]) -> Optional[UtilityValue]:
    return self.mapping[offer]
