"""Append-only, hash-chained experimental ledger (tamper-evident, like a blockchain without consensus).

Each entry commits to its predecessor: hash = SHA256(canonical JSON of entry without 'hash'), and
prev_hash = previous entry's hash. Editing, deleting or reordering any past entry breaks every later hash.
Artifacts (files) are recorded by SHA256 so a later change to a file is detectable by `verify(check_files=True)`.
Limits: a local file can be rewritten wholesale by someone who recomputes all hashes; the external anchor is
the git history (commit hash of the ledger is recorded in later entries / commit messages).
"""
import hashlib, json, os, subprocess, time

GENESIS = "0" * 64
KINDS = {"genesis", "decision", "implementation", "test", "run", "result", "cost", "gate", "amendment", "note"}


def _canon(d):
    return json.dumps(d, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def entry_hash(e):
    return hashlib.sha256(_canon({k: v for k, v in e.items() if k != "hash"}).encode()).hexdigest()


def read(path):
    if not os.path.exists(path): return []
    with open(path) as f: return [json.loads(l) for l in f if l.strip()]


def _git_head(root):
    try: return subprocess.check_output(["git", "-C", root, "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception: return None


def append(path, kind, summary, details=None, artifacts=(), root=".", backfilled=False, git_commit=None):
    """Add one block. `artifacts`: file paths (relative to root) hashed at append time."""
    assert kind in KINDS, kind
    chain = read(path)
    prev = chain[-1]["hash"] if chain else GENESIS
    e = dict(index=len(chain), utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), kind=kind, summary=summary,
             details=details or {}, artifacts={a: sha256_file(os.path.join(root, a)) for a in artifacts},
             git_commit=git_commit or _git_head(root), backfilled=backfilled, prev_hash=prev)
    e["hash"] = entry_hash(e)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a") as f: f.write(_canon(e) + "\n")
    return e


def verify(path, root=None):
    """Return (ok, problems). If root is given, also re-hash recorded artifacts against the files on disk
    (reports files that changed *since* they were logged; later legitimate edits are expected to be re-logged)."""
    chain, problems, prev = read(path), [], GENESIS
    for i, e in enumerate(chain):
        if e["index"] != i: problems.append(f"{i}: index mismatch")
        if e["prev_hash"] != prev: problems.append(f"{i}: prev_hash mismatch")
        if entry_hash(e) != e["hash"]: problems.append(f"{i}: content hash mismatch")
        if root:
            for a, h in e["artifacts"].items():
                p = os.path.join(root, a)
                if not os.path.exists(p): problems.append(f"{i}: artifact missing {a}")
                elif sha256_file(p) != h: problems.append(f"{i}: artifact changed since logging {a}")
        prev = e["hash"]
    return not problems, problems
