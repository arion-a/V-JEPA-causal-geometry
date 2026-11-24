"""CLI: ledger.py add KIND "summary" --title T --stage S --question Q --idea I --outcome O
[--detail k=v ...] [--artifact path ...] | verify [--files] | show"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from cg import ledger as L  # noqa: E402

PATH = ROOT / "log" / "ledger.jsonl"
FIELDS = ("title", "stage", "question", "idea", "outcome")


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    add = sub.add_parser("add")
    add.add_argument("kind", type=L.Kind)
    add.add_argument("summary")
    add.add_argument("--detail", action="append", default=[])
    add.add_argument("--artifact", action="append", default=[])
    add.add_argument("--removed", action="append", default=[], help="file deleted on purpose")
    for name in FIELDS:
        add.add_argument(
            f"--{name}", required=True, help=f"{name}: one short line, stored in the hashed entry"
        )
    verify = sub.add_parser("verify")
    verify.add_argument("--files", action="store_true")
    sub.add_parser("show")
    return ap


def main() -> int:
    n = parser().parse_args()
    if n.cmd == "add":
        details = dict(x.split("=", 1) for x in n.detail) | {f: getattr(n, f) for f in FIELDS}
        entry = L.append(
            PATH, L.Draft(n.kind, n.summary, details, n.artifact, removed=n.removed), ROOT
        )
        print(entry.index, entry.hash)
        return 0
    if n.cmd == "verify":
        ok, problems = L.verify(PATH, ROOT if n.files else None)
        print("OK" if ok else "BROKEN")
        for p in problems:
            print(" ", p)
        return 0 if ok else 1
    for e in L.read(PATH):
        print(e["index"], e["utc"], e["kind"], str(e["hash"])[:12], e["summary"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
