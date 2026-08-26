import json
import stat
import tarfile
from pathlib import Path

import pytest

from cms_tnp.condor import campaign_status, prepare_campaign, submit_campaign


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def _campaign_inputs(tmp_path):
    config = tmp_path / "photon.json"
    golden = tmp_path / "golden.json"
    environment = tmp_path / "environment.tar.gz"
    _write(golden, {"1": [[1, 10]]})
    _write(
        config,
        {
            "schema_version": 1,
            "profile": "photon_z",
            "measurement": "photon_id_sf",
            "year": "2025",
            "samples": {"data": ["data"], "mc": ["mc"]},
            "lumimask": "golden.json",
        },
    )
    with tarfile.open(environment, "w:gz"):
        pass
    for sample in ("data", "mc"):
        _write(
            tmp_path / f"{sample}_shards" / "shard_00000.json",
            {
                "schema_version": 1,
                "measurement": "photon_id_sf",
                "year": "2025",
                "sample": sample,
                "shard_id": 0,
                "records": [
                    {
                        "dataset": sample,
                        "sample": sample,
                        "file_path": f"root://example/{sample}.root",
                    }
                ],
            },
        )
    return config, environment


def test_condor_campaign_is_self_contained_and_output_gated(tmp_path):
    config, environment = _campaign_inputs(tmp_path)
    proxy = tmp_path / "proxy"
    proxy.write_text("not-a-real-credential")
    campaign = tmp_path / "campaign"
    summary = prepare_campaign(
        config_path=config,
        data_shards=tmp_path / "data_shards",
        mc_shards=tmp_path / "mc_shards",
        environment_path=environment,
        proxy_path=proxy,
        campaign_dir=campaign,
        job_flavour="workday",
    )
    assert summary == {
        "campaign_dir": str(campaign),
        "jobs": 2,
        "data_jobs": 1,
        "mc_jobs": 1,
    }
    assert stat.S_IMODE((campaign / "input" / "x509up").stat().st_mode) == 0o600
    assert "+JobFlavour" in (campaign / "submit.sub").read_text()
    with tarfile.open(campaign / "input" / "cms_tnp_inputs.tar.gz") as source:
        names = set(source.getnames())
    assert "config.json" in names
    assert "payloads/lumimask_golden.json" in names
    assert "shards/data_shard_00000.json" in names
    assert "shards/mc_shard_00000.json" in names

    initial = campaign_status(campaign)
    assert initial["status"] == "incomplete"
    assert len(initial["outputs_missing"]) == 2
    manifest = json.loads((campaign / "campaign.json").read_text())
    for job in manifest["jobs"]:
        _write(
            campaign / "outputs" / job["result"],
            {
                "sample": job["sample"],
                "status": "complete",
                "processing": {
                    "files_expected": job["files_expected"],
                    "files_processed": job["files_expected"],
                    "files_failed": [],
                },
            },
        )
    final = campaign_status(campaign)
    assert final["status"] == "complete"
    assert final["outputs_valid"] == 2


def test_condor_submission_records_cluster_and_blocks_duplicates(tmp_path):
    config, environment = _campaign_inputs(tmp_path)
    campaign = tmp_path / "campaign"
    prepare_campaign(
        config_path=config,
        data_shards=tmp_path / "data_shards",
        mc_shards=tmp_path / "mc_shards",
        environment_path=environment,
        campaign_dir=campaign,
    )
    command = tmp_path / "fake_condor_submit"
    command.write_text("#!/bin/sh\necho '2 job(s) submitted to cluster 12345.'\n")
    command.chmod(command.stat().st_mode | stat.S_IXUSR)
    result = submit_campaign(campaign, submit_command=str(command))
    assert result["cluster_id"] == 12345
    with pytest.raises(FileExistsError):
        submit_campaign(campaign, submit_command=str(command))


def test_condor_rejects_afs_paths(tmp_path):
    config, environment = _campaign_inputs(tmp_path)
    with pytest.raises(ValueError, match="AFS"):
        prepare_campaign(
            config_path=config,
            data_shards=tmp_path / "data_shards",
            mc_shards=tmp_path / "mc_shards",
            environment_path=environment,
            campaign_dir=tmp_path / "campaign",
            proxy_path=Path("/afs/example/x509up"),
        )
