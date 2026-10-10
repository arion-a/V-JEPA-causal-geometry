"""CLI: python scripts/ledger.py add KIND "summary" [--detail k=v ...] [--artifact path ...] | verify [--files] | show"""
import argparse, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from cg import ledger as L

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
PATH = os.path.join(ROOT, "log", "ledger.jsonl")
ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
a = sp.add_parser("add"); a.add_argument("kind"); a.add_argument("summary"); a.add_argument("--detail", action="append", default=[]); a.add_argument("--artifact", action="append", default=[])
v = sp.add_parser("verify"); v.add_argument("--files", action="store_true")
sp.add_parser("show")
n = ap.parse_args()
if n.cmd == "add":
    d = dict(x.split("=", 1) for x in n.detail)
    e = L.append(PATH, n.kind, n.summary, d, n.artifact, root=ROOT); print(e["index"], e["hash"])
elif n.cmd == "verify":
    ok, pr = L.verify(PATH, ROOT if n.files else None); print("OK" if ok else "BROKEN"); [print(" ", p) for p in pr]; sys.exit(0 if ok else 1)
else:
    for e in L.read(PATH): print(e["index"], e["utc"], e["kind"], e["hash"][:12], e["summary"])
