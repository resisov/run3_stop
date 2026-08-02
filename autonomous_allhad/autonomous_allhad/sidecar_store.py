#!/usr/bin/env python3
"""Indexed storage for flat-ntuple metadata sidecars.

The production worker historically wrote one large JSON document next to every
ROOT shard.  This module consolidates those documents into one SQLite database
without discarding their contents.  Readers keep backward compatibility with
adjacent JSON files and transparently fall back to the consolidated store.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import shutil
import sqlite3
import zlib
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable


STORE_NAME = "sidecars_main.sqlite"
STORE_SCHEMA = "flat_ntuple_sidecar_store_v1"


def root_key(path: Path) -> str:
    """Return a mount-point-independent key rooted below ``workflow``."""

    absolute = path.absolute()
    parts = absolute.parts
    workflow_positions = [index for index, part in enumerate(parts) if part == "workflow"]
    if not workflow_positions:
        raise ValueError(f"ROOT path is not below a workflow directory: {path}")
    index = workflow_positions[-1]
    return "/".join(parts[index + 1 :])


def workflow_directory(path: Path) -> Path:
    absolute = path.absolute()
    parts = absolute.parts
    workflow_positions = [index for index, part in enumerate(parts) if part == "workflow"]
    if not workflow_positions:
        raise ValueError(f"path is not below a workflow directory: {path}")
    index = workflow_positions[-1]
    return Path(*parts[: index + 1])


def store_path_for_root(path: Path) -> Path:
    configured = os.environ.get("AUTONOMOUS_ALLHAD_SIDECAR_STORE")
    if configured:
        return Path(configured)
    return workflow_directory(path) / STORE_NAME


@lru_cache(maxsize=8)
def _connection(path: str) -> sqlite3.Connection:
    uri = f"file:{Path(path).absolute()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    connection.execute("PRAGMA query_only = ON")
    return connection


def read_root_metadata(
    root_path: Path,
    *,
    fallback: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Read metadata from an adjacent JSON or the campaign main store."""

    adjacent = root_path.with_suffix(".json")
    if adjacent.is_file():
        payload = json.loads(adjacent.read_text())
        if not isinstance(payload, dict):
            raise RuntimeError(f"{adjacent}: expected a JSON object")
        return payload

    database = store_path_for_root(root_path)
    if database.is_file():
        row = _connection(str(database)).execute(
            "SELECT payload_zlib, sidecar_sha256 FROM sidecars WHERE root_key = ?",
            (root_key(root_path),),
        ).fetchone()
        if row is not None:
            raw = zlib.decompress(row[0])
            digest = hashlib.sha256(raw).hexdigest()
            if digest != row[1]:
                raise RuntimeError(
                    f"sidecar checksum mismatch for {root_path}: {digest} != {row[1]}"
                )
            payload = json.loads(raw)
            if not isinstance(payload, dict):
                raise RuntimeError(f"stored metadata for {root_path} is not an object")
            return payload

    if fallback is not None:
        return fallback
    raise FileNotFoundError(
        f"metadata absent from both {adjacent} and {database}: {root_path}"
    )


def _read_paths(path: Path) -> list[Path]:
    return [
        Path(line.strip())
        for line in path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _encode_sidecar(root: Path) -> tuple[str, int, int, str, str, str, bytes]:
    sidecar = root.with_suffix(".json")
    if not root.is_file() or not sidecar.is_file():
        raise FileNotFoundError(f"incomplete ROOT/JSON pair: {root}")
    raw = sidecar.read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise RuntimeError(f"{sidecar}: expected a JSON object")
    return (
        root_key(root),
        root.stat().st_size,
        len(raw),
        hashlib.sha256(raw).hexdigest(),
        str(payload.get("schema_version") or "unknown"),
        str(payload.get("status") or "unknown"),
        zlib.compress(raw, level=6),
    )


def consolidate(
    roots: Iterable[Path],
    output: Path,
    *,
    delete_sources: bool,
    workers: int,
) -> dict[str, Any]:
    roots = list(roots)
    if not roots:
        raise RuntimeError("no ROOT inputs were provided")
    keys = [root_key(path) for path in roots]
    if len(keys) != len(set(keys)):
        raise RuntimeError("duplicate canonical ROOT keys")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    if temporary.exists():
        temporary.unlink()
    connection = sqlite3.connect(temporary)
    try:
        connection.executescript(
            """
            PRAGMA journal_mode = DELETE;
            PRAGMA synchronous = FULL;
            CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE sidecars (
                root_key TEXT PRIMARY KEY,
                root_size INTEGER NOT NULL,
                sidecar_size INTEGER NOT NULL,
                sidecar_sha256 TEXT NOT NULL,
                schema_version TEXT,
                status TEXT,
                payload_zlib BLOB NOT NULL
            );
            CREATE INDEX sidecars_schema_status
                ON sidecars(schema_version, status);
            """
        )
        raw_bytes = 0
        compressed_bytes = 0
        status_counts: dict[str, int] = {}
        schema_counts: dict[str, int] = {}
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max(1, int(workers))
        ) as executor:
            records = executor.map(_encode_sidecar, roots)
            for index, record in enumerate(records, start=1):
                key, root_size, sidecar_size, digest, schema, status, compressed = record
                connection.execute(
                    "INSERT INTO sidecars VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        key,
                        root_size,
                        sidecar_size,
                        digest,
                        schema,
                        status,
                        compressed,
                    ),
                )
                raw_bytes += sidecar_size
                compressed_bytes += len(compressed)
                schema_counts[schema] = schema_counts.get(schema, 0) + 1
                status_counts[status] = status_counts.get(status, 0) + 1
                if index % 50 == 0:
                    connection.commit()
                    print(
                        json.dumps(
                            {
                                "phase": "write",
                                "completed": index,
                                "total": len(roots),
                                "raw_bytes": raw_bytes,
                                "compressed_bytes": compressed_bytes,
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
        summary = {
            "schema": STORE_SCHEMA,
            "status": "complete",
            "sidecars": len(roots),
            "raw_bytes": raw_bytes,
            "compressed_payload_bytes": compressed_bytes,
            "schema_counts": dict(sorted(schema_counts.items())),
            "status_counts": dict(sorted(status_counts.items())),
        }
        connection.executemany(
            "INSERT INTO metadata VALUES (?, ?)",
            [(key, json.dumps(value, sort_keys=True)) for key, value in summary.items()],
        )
        connection.commit()
    finally:
        connection.close()

    verify = sqlite3.connect(f"file:{temporary.absolute()}?mode=ro", uri=True)
    try:
        rows = verify.execute(
            "SELECT root_key, payload_zlib, sidecar_sha256 FROM sidecars ORDER BY root_key"
        )
        verified = 0
        for key, compressed, expected in rows:
            raw = zlib.decompress(compressed)
            if hashlib.sha256(raw).hexdigest() != expected:
                raise RuntimeError(f"verification checksum mismatch: {key}")
            payload = json.loads(raw)
            if not isinstance(payload, dict):
                raise RuntimeError(f"verification payload is not an object: {key}")
            verified += 1
        if verified != len(roots):
            raise RuntimeError(f"verification count mismatch: {verified} != {len(roots)}")
        integrity = verify.execute("PRAGMA integrity_check").fetchone()[0]
        if integrity != "ok":
            raise RuntimeError(f"SQLite integrity check failed: {integrity}")
    finally:
        verify.close()

    temporary.replace(output)
    database_sha256 = _sha256_file(output)
    deleted = 0
    if delete_sources:
        for root in roots:
            root.with_suffix(".json").unlink()
            deleted += 1
    result = {
        **summary,
        "output": str(output),
        "database_bytes": output.stat().st_size,
        "database_sha256": database_sha256,
        "verified": len(roots),
        "deleted_source_sidecars": deleted,
    }
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


def append_discovered(
    workflow: Path,
    store: Path,
    *,
    delete_sources: bool,
    workers: int,
) -> dict[str, Any]:
    """Append every remaining adjacent ROOT/JSON pair to an existing store."""

    roots = sorted(
        sidecar.with_suffix(".root")
        for sidecar in workflow.rglob("*.json")
        if sidecar.with_suffix(".root").is_file()
    )
    if not roots:
        raise RuntimeError(f"no adjacent ROOT/JSON pairs remain below {workflow}")
    keys = [root_key(path) for path in roots]
    if len(keys) != len(set(keys)):
        raise RuntimeError("duplicate discovered canonical ROOT keys")
    if not store.is_file():
        raise FileNotFoundError(store)

    temporary = store.with_name(f".{store.name}.extend.tmp.{os.getpid()}")
    if temporary.exists():
        temporary.unlink()
    shutil.copy2(store, temporary)
    connection = sqlite3.connect(temporary)
    try:
        existing = {
            row[0] for row in connection.execute("SELECT root_key FROM sidecars")
        }
        overlap = existing.intersection(keys)
        if overlap:
            raise RuntimeError(
                f"discovered sidecars already exist in the main store: {len(overlap)}"
            )
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max(1, int(workers))
        ) as executor:
            records = executor.map(_encode_sidecar, roots)
            for index, record in enumerate(records, start=1):
                connection.execute(
                    "INSERT INTO sidecars VALUES (?, ?, ?, ?, ?, ?, ?)", record
                )
                if index % 50 == 0:
                    connection.commit()
                    print(
                        json.dumps(
                            {
                                "phase": "append",
                                "completed": index,
                                "total": len(roots),
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
        connection.commit()
        total, raw_bytes, compressed_bytes = connection.execute(
            "SELECT count(*), sum(sidecar_size), sum(length(payload_zlib)) FROM sidecars"
        ).fetchone()
        schema_counts = dict(
            connection.execute(
                "SELECT schema_version, count(*) FROM sidecars GROUP BY schema_version"
            )
        )
        status_counts = dict(
            connection.execute(
                "SELECT status, count(*) FROM sidecars GROUP BY status"
            )
        )
        summary = {
            "schema": STORE_SCHEMA,
            "status": "complete",
            "sidecars": int(total),
            "raw_bytes": int(raw_bytes),
            "compressed_payload_bytes": int(compressed_bytes),
            "schema_counts": dict(sorted(schema_counts.items())),
            "status_counts": dict(sorted(status_counts.items())),
        }
        connection.execute("DELETE FROM metadata")
        connection.executemany(
            "INSERT INTO metadata VALUES (?, ?)",
            [(key, json.dumps(value, sort_keys=True)) for key, value in summary.items()],
        )
        connection.commit()
    finally:
        connection.close()

    verify = sqlite3.connect(f"file:{temporary.absolute()}?mode=ro", uri=True)
    try:
        verified = 0
        wanted = set(keys)
        for key, compressed, expected in verify.execute(
            "SELECT root_key, payload_zlib, sidecar_sha256 "
            "FROM sidecars ORDER BY root_key"
        ):
            if key not in wanted:
                continue
            raw = zlib.decompress(compressed)
            if hashlib.sha256(raw).hexdigest() != expected:
                raise RuntimeError(f"new sidecar checksum mismatch: {key}")
            if not isinstance(json.loads(raw), dict):
                raise RuntimeError(f"new sidecar payload is not an object: {key}")
            verified += 1
        if verified != len(keys):
            raise RuntimeError(
                f"new sidecar verification count mismatch: {verified} != {len(keys)}"
            )
        integrity = verify.execute("PRAGMA integrity_check").fetchone()[0]
        if integrity != "ok":
            raise RuntimeError(f"SQLite integrity check failed: {integrity}")
    finally:
        verify.close()

    temporary.replace(store)
    database_sha256 = _sha256_file(store)
    deleted = 0
    if delete_sources:
        for root in roots:
            root.with_suffix(".json").unlink()
            deleted += 1
    result = {
        **summary,
        "output": str(store),
        "database_bytes": store.stat().st_size,
        "database_sha256": database_sha256,
        "appended": len(roots),
        "verified_appended": verified,
        "deleted_source_sidecars": deleted,
    }
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("consolidate")
    build.add_argument("--root-list", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--delete-sources", action="store_true")
    build.add_argument("--workers", type=int, default=8)
    append = subparsers.add_parser("append-discovered")
    append.add_argument("--workflow", type=Path, required=True)
    append.add_argument("--store", type=Path, required=True)
    append.add_argument("--delete-sources", action="store_true")
    append.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)
    if args.command == "consolidate":
        consolidate(
            _read_paths(args.root_list),
            args.output,
            delete_sources=bool(args.delete_sources),
            workers=max(1, int(args.workers)),
        )
    elif args.command == "append-discovered":
        append_discovered(
            args.workflow,
            args.store,
            delete_sources=bool(args.delete_sources),
            workers=max(1, int(args.workers)),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
