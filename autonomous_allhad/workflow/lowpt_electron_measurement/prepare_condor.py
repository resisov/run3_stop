#!/usr/bin/env python3
"""Prepare the EOS-only low-pT electron TnP Condor campaign."""
from workflow.prepare_tnp_measurement_condor import cli


if __name__ == "__main__":
    raise SystemExit(cli(default_kind="electron"))
