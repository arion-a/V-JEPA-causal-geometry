import json
from cg import ledger as L


def test_chain_and_tamper_detection(tmp_path):
    p = str(tmp_path / "l.jsonl"); (tmp_path / "f.txt").write_text("a")
    L.append(p, "genesis", "start", root=str(tmp_path))
    L.append(p, "implementation", "file", artifacts=["f.txt"], root=str(tmp_path))
    L.append(p, "run", "third", root=str(tmp_path))
    assert L.verify(p, str(tmp_path)) == (True, [])
    # edit a past entry's content
    rows = [json.loads(x) for x in open(p)]; rows[1]["summary"] = "changed"
    open(p, "w").write("\n".join(json.dumps(r) for r in rows) + "\n")
    ok, pr = L.verify(p); assert not ok and any("1:" in x for x in pr)
    # deleting an entry breaks the chain
    rows = [json.loads(x) for x in open(p)]; rows[1]["summary"] = "file"
    del rows[1]; open(p, "w").write("\n".join(json.dumps(r) for r in rows) + "\n")
    assert not L.verify(p)[0]


def test_artifact_change_detected(tmp_path):
    p = str(tmp_path / "l.jsonl"); f = tmp_path / "f.txt"; f.write_text("a")
    L.append(p, "implementation", "x", artifacts=["f.txt"], root=str(tmp_path))
    f.write_text("b")
    ok, pr = L.verify(p, str(tmp_path)); assert not ok and "changed" in pr[0]
