import json
from pathlib import Path

import pytest

from cg import ledger as lg


def read_rows(p: Path) -> list[dict[str, object]]:
    return [json.loads(x) for x in p.read_text().splitlines()]


def write_rows(p: Path, rows: list[dict[str, object]]) -> None:
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def test_chain_and_tamper_detection(tmp_path: Path) -> None:
    p = tmp_path / "l.jsonl"
    (tmp_path / "f.txt").write_text("a")
    lg.append(p, lg.Draft(lg.Kind.GENESIS, "start"), tmp_path)
    lg.append(p, lg.Draft(lg.Kind.IMPLEMENTATION, "file", artifacts=["f.txt"]), tmp_path)
    lg.append(p, lg.Draft(lg.Kind.RUN, "third"), tmp_path)
    assert lg.verify(p, tmp_path) == (True, [])
    rows = read_rows(p)
    rows[1]["summary"] = "changed"
    write_rows(p, rows)
    ok, problems = lg.verify(p)
    assert not ok
    assert any("1:" in x for x in problems)
    rows[1]["summary"] = "file"
    del rows[1]
    write_rows(p, rows)
    assert not lg.verify(p).ok


def test_an_added_field_is_detected_not_silently_dropped(tmp_path: Path) -> None:
    p = tmp_path / "l.jsonl"
    lg.append(p, lg.Draft(lg.Kind.NOTE, "x"), tmp_path)
    rows = read_rows(p)
    rows[0]["extra"] = "smuggled"
    write_rows(p, rows)
    assert "0: content hash mismatch" in lg.verify(p).problems


def test_artifact_change_detected(tmp_path: Path) -> None:
    p, f = tmp_path / "l.jsonl", tmp_path / "f.txt"
    f.write_text("a")
    lg.append(p, lg.Draft(lg.Kind.IMPLEMENTATION, "x", artifacts=["f.txt"]), tmp_path)
    f.write_text("b")
    ok, problems = lg.verify(p, tmp_path)
    assert not ok
    assert "changed" in problems[0]


def test_relogged_artifact_supersedes_old_hash(tmp_path: Path) -> None:
    p, f = tmp_path / "l.jsonl", tmp_path / "f.txt"
    f.write_text("a")
    lg.append(p, lg.Draft(lg.Kind.IMPLEMENTATION, "v1", artifacts=["f.txt"]), tmp_path)
    f.write_text("b")
    lg.append(p, lg.Draft(lg.Kind.IMPLEMENTATION, "v2", artifacts=["f.txt"]), tmp_path)
    assert lg.verify(p, tmp_path).ok
    f.write_text("c")
    assert not lg.verify(p, tmp_path).ok


def test_unknown_kind_and_missing_artifact_fail_loudly(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        lg.Kind("opinion")
    with pytest.raises(FileNotFoundError):
        lg.append(
            tmp_path / "l.jsonl", lg.Draft(lg.Kind.NOTE, "x", artifacts=["nope.txt"]), tmp_path
        )


def test_a_file_removed_on_purpose_is_logged_and_verified(tmp_path: Path) -> None:
    p, f = tmp_path / "l.jsonl", tmp_path / "f.txt"
    f.write_text("a")
    lg.append(p, lg.Draft(lg.Kind.IMPLEMENTATION, "add", artifacts=["f.txt"]), tmp_path)
    f.unlink()
    assert not lg.verify(p, tmp_path).ok
    lg.append(p, lg.Draft(lg.Kind.NOTE, "delete", removed=["f.txt"]), tmp_path)
    assert lg.verify(p, tmp_path).ok
    f.write_text("back")
    assert "still exists" in lg.verify(p, tmp_path).problems[0]


def test_removal_cannot_be_logged_for_an_existing_file(tmp_path: Path) -> None:
    (tmp_path / "f.txt").write_text("a")
    with pytest.raises(ValueError):
        lg.append(tmp_path / "l.jsonl", lg.Draft(lg.Kind.NOTE, "x", removed=["f.txt"]), tmp_path)
