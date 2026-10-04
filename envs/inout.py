from __future__ import annotations

import xml.etree.ElementTree as ET
from os import PathLike, listdir
from pathlib import Path
from typing import Iterable

from negmas.outcomes.outcome_space import make_os

from negmas.outcomes import issues_from_genius
from negmas.preferences import (
    UtilityFunction,
    make_discounted_ufun,
)
from negmas.sao import SAOMechanism
from negmas.inout import Scenario

def load_genius_domain(
    domain_file_name: PathLike,
    utility_file_names: Iterable[PathLike] | None = None,
    ignore_discount=False,
    ignore_reserved=False,
    safe_parsing=True,
    mechanism_type=SAOMechanism,
    **kwargs,
) -> Scenario:

    issues = None
    if domain_file_name is not None:
        issues, _ = issues_from_genius(
            domain_file_name,
            safe_parsing=safe_parsing,
        )

    agent_info = []
    if utility_file_names is None:
        utility_file_names = []
    for ufname in utility_file_names:
        utility, discount_factor = UtilityFunction.from_genius(
            file_name=ufname,
            issues=issues,
            safe_parsing=safe_parsing,
            ignore_discount=ignore_discount,
            ignore_reserved=ignore_reserved,
            name=str(ufname),
        )
        agent_info.append(
            {
                "ufun": utility,
                "ufun_name": ufname,
                "reserved_value_func": utility.reserved_value
                if utility is not None
                else float("-inf"),
                "discount_factor": discount_factor,
            }
        )
    if domain_file_name is not None:
        kwargs["avoid_ultimatum"] = False
        kwargs["dynamic_entry"] = False
        kwargs["max_n_agents"] = None
        if not ignore_discount:
            for info in agent_info:
                info["ufun"] = (
                    info["ufun"]
                    if info["discount_factor"] is None or info["discount_factor"] == 1.0
                    else make_discounted_ufun(
                        ufun=info["ufun"],
                        discount_per_round=info["discount_factor"],
                        power_per_round=1.0,
                        # reserved_value=info["reserved_value_func"],
                    )
                )
    if issues is None:
        raise ValueError(f"Could not load domain {domain_file_name}")

    return Scenario(
        agenda=make_os(issues, name=str(domain_file_name)),
        ufuns=[_["ufun"] for _ in agent_info],  # type: ignore
        mechanism_type=mechanism_type,
        mechanism_params=kwargs,
    )

def load_genius_domain_from_folder(
    folder_name: str | PathLike,
    ignore_reserved=False,
    ignore_discount=False,
    safe_parsing=False,
    mechanism_type=SAOMechanism,
    **kwargs,
) -> Scenario:
    folder_name = str(folder_name)
    files = sorted(listdir(folder_name))
    domain_file_name = None
    utility_file_names = []
    for f in files:
        if not f.endswith(".xml") or f.endswith("pareto.xml"):
            continue
        full_name = folder_name + "/" + f
        root = ET.parse(full_name).getroot()

        if root.tag == "negotiation_template":
            domain_file_name = Path(full_name)
        elif root.tag == "utility_space":
            utility_file_names.append(full_name)
    if domain_file_name is None:
        raise ValueError("Cannot find a domain file")
    return load_genius_domain(
        domain_file_name=domain_file_name,
        utility_file_names=utility_file_names,
        safe_parsing=safe_parsing,
        ignore_reserved=ignore_reserved,
        ignore_discount=ignore_discount,
        mechanism_type=mechanism_type,
        **kwargs,
    )