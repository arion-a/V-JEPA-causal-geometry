"""Append-only, hash-chained experimental ledger (tamper-evident, no consensus).

Each entry commits to its predecessor: hash = SHA256(canonical JSON of the entry without 'hash')
and prev_hash = the previous entry's hash, so editing, deleting or reordering an entry breaks
every later hash. Artifacts (files) are recorded by SHA256 so a later change to a file is
detectable. A local file can be rewritten wholesale by someone who recomputes every hash; the
external anchor is the git history.

The functional core (canonical, build_entry, verify_chain) takes and returns plain values.
Reading and writing files, the clock and git live in read, append and verify at the bottom.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import NamedTuple

from cg.json_types import JsonValue

GENESIS = "0" * 64
REMOVED = "removed"
type RawEntry = dict[str, JsonValue]


class Kind(StrEnum):
    GENESIS = "genesis"
    DECISION = "decision"
    IMPLEMENTATION = "implementation"
    TEST = "test"
    RUN = "run"
    RESULT = "result"
    COST = "cost"
    GATE = "gate"
    AMENDMENT = "amendment"
    NOTE = "note"


@dataclass(frozen=True)
class Draft:
    """What the author supplies; ``artifacts`` are file paths relative to the repository root."""

    kind: Kind
    summary: str
    details: Mapping[str, JsonValue] = field(default_factory=dict)
    artifacts: Sequence[str] = ()
    backfilled: bool = False
    removed: Sequence[str] = ()


@dataclass(frozen=True)
class Link:
    prev_hash: str
    index: int
    utc: str


@dataclass(frozen=True)
class Entry:
    index: int
    utc: str
    kind: Kind
    summary: str
    details: Mapping[str, JsonValue]
    artifacts: Mapping[str, str]
    git_commit: str | None
    backfilled: bool
    prev_hash: str
    hash: str


class Verification(NamedTuple):
    ok: bool
    problems: list[str]


def canonical(obj: JsonValue) -> str:
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def entry_hash(entry: Mapping[str, JsonValue]) -> str:
    body: JsonValue = {k: v for k, v in entry.items() if k != "hash"}
    return hashlib.sha256(canonical(body).encode()).hexdigest()


def build_entry(
    link: Link, draft: Draft, artifact_hashes: Mapping[str, str], git_commit: str | None
) -> Entry:
    body: RawEntry = {
        "index": link.index,
        "utc": link.utc,
        "kind": draft.kind.value,
        "summary": draft.summary,
        "details": dict(draft.details),
        "artifacts": dict(artifact_hashes),
        "git_commit": git_commit,
        "backfilled": draft.backfilled,
        "prev_hash": link.prev_hash,
    }
    return Entry(
        link.index,
        link.utc,
        draft.kind,
        draft.summary,
        draft.details,
        artifact_hashes,
        git_commit,
        draft.backfilled,
        link.prev_hash,
        entry_hash(body),
    )


def to_json(entry: Entry) -> RawEntry:
    return {
        "index": entry.index,
        "utc": entry.utc,
        "kind": entry.kind.value,
        "summary": entry.summary,
        "details": dict(entry.details),
        "artifacts": dict(entry.artifacts),
        "git_commit": entry.git_commit,
        "backfilled": entry.backfilled,
        "prev_hash": entry.prev_hash,
        "hash": entry.hash,
    }


def _artifacts_of(entry: Mapping[str, JsonValue]) -> dict[str, str]:
    raw = entry.get("artifacts")
    if not isinstance(raw, dict):
        return {}
    return {name: h for name, h in raw.items() if isinstance(h, str)}


def _link_problems(i: int, entry: Mapping[str, JsonValue], prev: str) -> list[str]:
    checks = [
        (entry.get("index") != i, "index mismatch"),
        (entry.get("prev_hash") != prev, "prev_hash mismatch"),
        (entry_hash(entry) != entry.get("hash"), "content hash mismatch"),
    ]
    return [f"{i}: {message}" for failed, message in checks if failed]


def _file_problems(
    i: int, artifacts: Mapping[str, str], disk: Mapping[str, str | None]
) -> list[str]:
    problems = []
    for name, logged in artifacts.items():
        actual = disk[name]
        if logged == REMOVED:
            if actual is not None:
                problems.append(f"{i}: artifact logged as removed still exists {name}")
        elif actual is None:
            problems.append(f"{i}: artifact missing {name}")
        elif actual != logged:
            problems.append(f"{i}: artifact changed since logging {name}")
    return problems


def verify_chain(
    chain: Sequence[Mapping[str, JsonValue]], disk: Mapping[str, str | None] | None
) -> Verification:
    """Check links and hashes; with ``disk`` (artifact name -> current SHA256 or None if missing)
    also check each file against its newest logged hash only, since later logs supersede earlier."""
    latest = {name: i for i, e in enumerate(chain) for name in _artifacts_of(e)}
    problems: list[str] = []
    prev = GENESIS
    for i, entry in enumerate(chain):
        problems += _link_problems(i, entry, prev)
        if disk is not None:
            newest = {n: h for n, h in _artifacts_of(entry).items() if latest[n] == i}
            problems += _file_problems(i, newest, disk)
        prev = str(entry.get("hash"))
    return Verification(not problems, problems)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read(path: Path) -> list[RawEntry]:
    if not path.exists():
        return []
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _git_head(root: Path) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.strip()


def append(
    path: Path,
    draft: Draft,
    root: Path = Path("."),
    *,
    git_commit: str | None = None,
    clock: Callable[[], str] = _now_utc,
) -> Entry:
    """Add one block, hashing the draft's artifacts now."""
    chain = read(path)
    link = Link(str(chain[-1]["hash"]) if chain else GENESIS, len(chain), clock())
    if present := [r for r in draft.removed if (root / r).exists()]:
        raise ValueError(f"cannot log files as removed while they exist: {present}")
    hashes = {a: sha256_file(root / a) for a in draft.artifacts} | dict.fromkeys(
        draft.removed, REMOVED
    )
    entry = build_entry(link, draft, hashes, git_commit or _git_head(root))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(canonical(to_json(entry)) + "\n")
    return entry


def verify(path: Path, root: Path | None = None) -> Verification:
    """Verify the chain; with ``root`` also re-hash the recorded artifacts on disk."""
    chain = read(path)
    if root is None:
        return verify_chain(chain, None)
    names = {n for e in chain for n in _artifacts_of(e)}
    disk = {n: sha256_file(root / n) if (root / n).exists() else None for n in names}
    return verify_chain(chain, disk)
