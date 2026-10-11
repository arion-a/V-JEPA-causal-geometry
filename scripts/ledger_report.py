"""Build reports/ledger_live.html: a self-contained page showing every ledger node. Re-verifies the hash chain in the browser (SHA-256).
Usage: python3 scripts/ledger_report.py   (then republish the page as the Artifact so the viewer sees new nodes)."""
import json, os, sys, time
ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")); sys.path.insert(0, ROOT)
from cg import ledger as L
chain = L.read(f"{ROOT}/log/ledger.jsonl"); ok, problems = L.verify(f"{ROOT}/log/ledger.jsonl", root=ROOT)
titles = json.load(open(f"{ROOT}/log/titles.json"))
for e in chain: e["title"] = e["details"].get("title") or titles.get(str(e["index"]), "")      # display only: new nodes carry the title inside the hashed entry; nodes 0-70 use log/titles.json
data = json.dumps(dict(chain=chain, server_ok=ok, server_problems=problems, built=time.strftime("%Y-%m-%d %H:%M:%SZ", time.gmtime())), ensure_ascii=False).replace("</", "<\\/")
html = open(f"{ROOT}/scripts/ledger_report_template.html").read().replace("/*__DATA__*/null", data)
open(f"{ROOT}/reports/ledger_live.html", "w").write(html); print("nodes:", len(chain), "server verify:", ok)
