#!/usr/bin/env python3
"""Merge all low-pT muon TnP histogram shard JSONs."""
from workflow.tnp_measurement_reduce import cli


if __name__ == "__main__":
    raise SystemExit(cli(default_kind="muon"))
