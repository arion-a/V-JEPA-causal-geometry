import json
from pathlib import Path

from golden_util import load, subtree

from cg import ledger as lg
from cg.json_types import JsonValue

BASE = load("ledger_golden")
OUT = subtree(BASE, "out")
SPECS: list[tuple[lg.Kind, str, dict[str, JsonValue], bool]] = [
    (lg.Kind.GENESIS, "open", {}, False),
    (lg.Kind.DECISION, "pick", {"title": "t", "n": 3, "nested": {"a": [1, 2.5]}}, False),
    (lg.Kind.NOTE, "uni éè", {}, True),
]


def strings(prefix: str) -> list[str]:
    items = subtree(OUT, prefix)
    items.pop("", None)
    return [str(items[f"/{i}"]) for i in range(len(items))]


def build(tmp: Path) -> Path:
    (tmp / "a.txt").write_text("hello")
    path = tmp / "log" / "l.jsonl"
    ticks = iter(["2026-01-01T00:00:00Z", "2026-01-01T00:00:01Z", "2026-01-01T00:00:02Z"])
    for kind, summary, details, backfilled in SPECS:
        arts = ["a.txt"] if kind is lg.Kind.DECISION else []
        draft = lg.Draft(kind, summary, details, arts, backfilled)
        lg.append(path, draft, tmp, git_commit="deadbeef", clock=lambda: next(ticks))
    return path


def test_written_chain_is_byte_identical_to_pre_refactor_output(tmp_path: Path) -> None:
    path = build(tmp_path)
    assert path.read_text().splitlines() == strings("/lines")
    chain = lg.read(path)
    assert lg.entry_hash(chain[0]) == str(OUT["/hash0"])
    assert lg.canonical({"b": 1, "a": [1, 2], "c": "é"}) == str(OUT["/canon"])


def test_verification_results_match_pre_refactor_output(tmp_path: Path) -> None:
    path = build(tmp_path)
    assert lg.verify(path, tmp_path) == (bool(OUT["/ok"]), strings("/probs"))
    (tmp_path / "a.txt").write_text("changed")
    assert lg.verify(path, tmp_path) == (bool(OUT["/ok2"]), strings("/probs2"))
    rows = [json.loads(x) for x in path.read_text().splitlines()]
    rows[1]["summary"] = "forged"
    tampered = tmp_path / "t.jsonl"
    tampered.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    assert lg.verify(tampered) == (bool(OUT["/ok3"]), strings("/probs3"))
    del rows[0]
    tampered.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    assert lg.verify(tampered) == (bool(OUT["/ok4"]), strings("/probs4"))
    (tmp_path / "a.txt").unlink()
    assert lg.verify(path, tmp_path) == (bool(OUT["/ok5"]), strings("/probs5"))
