from __future__ import annotations

import re


SUPPORTED_SIGNAL_TOPOLOGIES = ("T2tt", "T2tb", "T2bW")
SIGNAL_TOPOLOGY_IDS = {
    "": 0,
    "T2tt": 1,
    "T2tb": 2,
    "T2bW": 3,
}


def signal_genmodel_branch(name: str) -> bool:
    text = str(name)
    return any(text.startswith(f"GenModel_{topology}_") for topology in SUPPORTED_SIGNAL_TOPOLOGIES)


def signal_runs_sumw_branch(name: str) -> bool:
    text = str(name)
    return any(text.startswith(f"genEventSumw_{topology}_") for topology in SUPPORTED_SIGNAL_TOPOLOGIES)


def signal_topology(name: str) -> str:
    text = str(name)
    for topology in SUPPORTED_SIGNAL_TOPOLOGIES:
        if (
            f"GenModel_{topology}_" in text
            or f"genEventSumw_{topology}_" in text
            or f"SMS-2Stop-{topology}_" in text
        ):
            return topology
    # The original Run-3 FastSim production omitted "T2tt" from its dataset
    # primary name. Its event-level model branch remains authoritative.
    if "SMS-2Stop_Par-mStop-" in text:
        return "T2tt"
    return ""


def signal_topology_id(name: str) -> int:
    return SIGNAL_TOPOLOGY_IDS[signal_topology(name)]


def signal_mass_from_genmodel(name: str) -> tuple[str, int | None, int | None]:
    topology = signal_topology(name)
    numbers = re.findall(r"(\d+)", str(name))
    if not topology or len(numbers) < 2:
        return topology, None, None
    return topology, int(numbers[-2]), int(numbers[-1])


def signal_mass_key(topology: str, mstop: int, mlsp: int) -> str:
    # Preserve the established T2tt key contract while keeping the two new
    # interpretations disjoint in shared normalization/histogram payloads.
    prefix = "" if topology == "T2tt" else f"{topology}_"
    return f"{prefix}mStop{int(mstop)}_mLSP{int(mlsp)}"
