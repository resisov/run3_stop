#!/usr/bin/env python3
"""Count one low-pT muon TnP shard directly to histogram JSON."""
from workflow.tnp_measurement_shard import cli


if __name__ == "__main__":
    raise SystemExit(cli(default_kind="muon"))
