#!/usr/bin/env python3
"""Attach the validated 2024 MC counts to the completed 2025 data measurement."""
import json
from pathlib import Path


def main() -> int:
    output = Path("autonomous_allhad/outputs/trigger_efficiency/2025/flat_quick/counts.json")
    mc_source = Path("autonomous_allhad/outputs/trigger_efficiency/2024/flat_quick/counts.json")
    data = json.loads(output.read_text())
    mc = json.loads(mc_source.read_text())
    if data["bin_edges_gev"] != mc["bin_edges_gev"]:
        raise RuntimeError("2024 MC and 2025 data binning differ")
    data["measurement"] = "met_or_2025_data_with_2024_mc"
    data["mc"] = mc["mc"]
    data["mc_source"] = str(mc_source)
    data["status"] = "preliminary_2025_data_with_2024_mc_not_for_adoption"
    output.write_text(json.dumps(data, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
