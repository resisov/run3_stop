from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from ..paths import PathKind, PathPolicy, PathPolicyError


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS chunks (
  chunk_id TEXT PRIMARY KEY,
  dataset TEXT NOT NULL,
  input_files TEXT NOT NULL,
  entry_start INTEGER,
  entry_stop INTEGER,
  shift TEXT NOT NULL DEFAULT 'nominal',
  state TEXT NOT NULL,
  retry_count INTEGER NOT NULL DEFAULT 0,
  cluster_id TEXT,
  process_id TEXT,
  start_time TEXT,
  end_time TEXT,
  output_path TEXT,
  output_size INTEGER,
  checksum TEXT,
  validation_status TEXT,
  error_category TEXT,
  software_version TEXT,
  schema_version TEXT
);
"""

COMPLETE_STATES = {"validated"}


@dataclass
class ChunkRecord:
    chunk_id: str
    dataset: str
    input_files: list
    state: str = "planned"
    output_path: object = None
    validation_status: object = None


def initialize(path, dry_run=False):
    policy = PathPolicy.default()
    db_path = policy.resolve(path, PathKind.OUTPUT)
    if dry_run:
        return db_path
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as con:
        con.executescript(SCHEMA_SQL)
    return db_path


def export_json(db_path):
    policy = PathPolicy.default()
    resolved = policy.resolve(db_path, PathKind.INPUT)
    with sqlite3.connect(resolved) as con:
        rows = con.execute("SELECT * FROM chunks").fetchall()
        names = [desc[0] for desc in con.execute("SELECT * FROM chunks LIMIT 0").description]
    return {"chunks": [dict(zip(names, row)) for row in rows]}


def load_benchmark_manifest(path):
    policy = PathPolicy.default()
    resolved = policy.resolve(path, PathKind.INPUT)
    with resolved.open() as handle:
        manifest = json.load(handle)
    listed_urls = set(manifest.get("remote_urls", []))
    for sample in manifest.get("samples", []):
        for input_path in sample.get("files", []):
            if str(input_path).startswith(("root://", "xrootd://")):
                policy.validate_remote_url(input_path, listed_urls)
            else:
                policy.resolve(input_path, PathKind.INPUT)
    return manifest


def is_chunk_complete(record, policy=None):
    policy = policy or PathPolicy.default()
    if record.state not in COMPLETE_STATES or record.validation_status != "passed":
        return False
    if not record.output_path:
        return False
    try:
        output = policy.resolve(record.output_path, PathKind.INPUT)
    except PathPolicyError:
        return False
    return output.is_file() and output.stat().st_size > 0
