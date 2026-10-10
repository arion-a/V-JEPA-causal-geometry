#!/usr/bin/env python3
"""Git-based pod worker. Run once inside the pod; no ports are opened.

Loop: git pull -> run every not-yet-run `jobs/NNN_name.sh` in order -> commit+push `results/jobs/<name>/` -> sleep.
A job is 'done' when results/jobs/<name>/status.json exists in the repo (so reruns need a new job name).
Job env: JOB_OUT (small files to commit, keep < 1 MB each), CG_ROOT (persistent volume; big files stay there).
Job header comment `# timeout: SECONDS` sets its limit (default 1800, hard cap 7200).
Safety: GitHub token is read from $GH_TOKEN or a hidden prompt, passed to git via a credential helper that reads
the env var (never written to .git/config or logs) and is REMOVED from the job's environment.
Kill switch: commit a file `jobs/STOP` (or Ctrl-C). Anyone who can push to the repo can run code here:
use a token scoped to this one repo and keep repo write access to yourself.
"""
import argparse, getpass, hashlib, json, os, re, shutil, subprocess, sys, time

REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
MAX_TIMEOUT, DEFAULT_TIMEOUT, MAX_FILE = 7200, 1800, 1_000_000
CRED = '!f(){ echo username=x-access-token; echo "password=$GH_TOKEN"; }; f'


def clean_token(t):
    """Terminal pastes can carry CR/LF/spaces; git refuses a credential containing them. Strip all whitespace/control chars."""
    return "".join(ch for ch in (t or "") if ch.isprintable() and not ch.isspace())


TOKEN_FILE = os.environ.get("CG_TOKEN_FILE", os.path.join(os.environ.get("CG_ROOT", "/workspace/vjepa-causal-geometry"), "secrets", "gh_token"))


def describe_token(t):
    """Safe diagnosis (never returns the token): length, prefix class, hint."""
    t = t or ""; ok_prefix = t.startswith(("github_pat_", "ghp_"))
    hint = ("EMPTY - the hidden paste went in empty; paste again (Ctrl+Shift+V) and press Enter" if not t else
            "does not start with github_pat_/ghp_ - the wrong string was copied" if not ok_prefix else
            "unusual length (expect ~93 for github_pat_, 40 for ghp_) - re-copy with GitHub's copy button" if (t.startswith("github_pat_") and not 80 <= len(t) <= 110) else "looks like a GitHub token")
    return dict(length=len(t), prefix=t[:11] if ok_prefix else "(not github_pat_/ghp_)", hint=hint)


def load_token():
    """GH_TOKEN env > token file (chmod 600 on the persistent volume) > hidden prompt. Always whitespace-cleaned."""
    t = clean_token(os.environ.get("GH_TOKEN", ""))
    if not t and os.path.exists(TOKEN_FILE): t = clean_token(open(TOKEN_FILE).read())
    return t


def auth_check():
    """Read access check with the current token (ls-remote). Returns (ok, message). Push rights are only proven by a real push."""
    p = git("ls-remote", "origin", "HEAD", check=False)
    return p.returncode == 0, (p.stderr.strip()[-200:] if p.returncode else "ok")


def git(*a, check=True, token=True):
    cmd = ["git", "-C", REPO] + (["-c", f"credential.helper={CRED}"] if token and os.environ.get("GH_TOKEN") else []) + list(a)
    p = subprocess.run(cmd, capture_output=True, text=True)
    if check and p.returncode: raise RuntimeError(f"git {a[0]} failed: {p.stderr.strip()[-400:]}")
    return p


def pending():
    jd = os.path.join(REPO, "jobs")
    names = sorted(f for f in os.listdir(jd) if re.fullmatch(r"\d{3}_[\w.-]+\.sh", f)) if os.path.isdir(jd) else []
    return [n for n in names if not os.path.exists(os.path.join(REPO, "results", "jobs", n[:-3], "status.json"))]


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def run_job(name):
    path, stem = os.path.join(REPO, "jobs", name), name[:-3]
    out = os.path.join(REPO, "results", "jobs", stem); os.makedirs(out, exist_ok=True)
    head = open(path).read(2000)
    m = re.search(r"#\s*timeout:\s*(\d+)", head); timeout = min(int(m.group(1)) if m else DEFAULT_TIMEOUT, MAX_TIMEOUT)
    env = {k: v for k, v in os.environ.items() if k not in ("GH_TOKEN", "GITHUB_TOKEN")}
    env.update(JOB_OUT=out, CG_ROOT=os.environ.get("CG_ROOT", "/workspace/vjepa-causal-geometry"))
    t0 = time.time(); started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    try:
        p = subprocess.run(["bash", path], cwd=REPO, env=env, capture_output=True, text=True, timeout=timeout)
        rc, log = p.returncode, p.stdout + p.stderr
    except subprocess.TimeoutExpired as e:
        rc, log = -9, f"TIMEOUT after {timeout}s\n" + str(e.stdout or "")[-20000:]
    open(os.path.join(out, "log.txt"), "w").write(log[-200_000:])
    gpu = subprocess.run("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader", shell=True, capture_output=True, text=True).stdout.strip()
    json.dump(dict(job=name, rc=rc, started=started, seconds=round(time.time() - t0, 1), timeout=timeout, job_sha256=sha(path),
                   host=os.uname().nodename, gpu=gpu, commit=git("rev-parse", "HEAD").stdout.strip()), open(os.path.join(out, "status.json"), "w"), indent=1)
    for root, _, fs in os.walk(out):                      # never commit big files (they stay on the volume)
        for f in fs:
            p_ = os.path.join(root, f)
            if os.path.getsize(p_) > MAX_FILE:
                open(p_ + ".TOO_LARGE", "w").write(f"{os.path.getsize(p_)} bytes, sha256 {sha(p_)}; kept on volume\n"); os.remove(p_)
    return rc


def publish(stem):
    git("add", "-f", f"results/jobs/{stem}")
    git("-c", "user.name=pod-worker", "-c", "user.email=pod-worker@users.noreply.github.com", "commit", "-m", f"pod-worker: results for {stem}")
    return push_pending()


def unpushed():
    p = git("rev-list", "--count", "origin/main..HEAD", check=False)
    return int(p.stdout.strip() or 0) if p.returncode == 0 else 0


def push_pending():
    """Push local commits, retrying with rebase; prints git's error (stderr never contains the token)."""
    if not unpushed(): return True
    err = ""
    for _ in range(4):
        p = git("push", "origin", "HEAD:main", check=False)
        if p.returncode == 0: return True
        err = p.stderr.strip()[-300:]
        git("pull", "--rebase", "origin", "main", check=False); time.sleep(2)
    print(f"[worker] PUSH FAILED ({unpushed()} commit(s) waiting): {err}", flush=True)
    return False


def publish_orphans():
    """Jobs that ran but whose results were never committed (e.g. an earlier publish failed): publish, do not rerun."""
    jd = os.path.join(REPO, "results", "jobs")
    for stem in sorted(os.listdir(jd)) if os.path.isdir(jd) else []:
        if os.path.exists(os.path.join(jd, stem, "status.json")) and not git("ls-files", f"results/jobs/{stem}/status.json").stdout.strip():
            print(f"[worker] publishing unpublished results for {stem}", flush=True); publish(stem)


def cycle():
    p = git("pull", "--ff-only", check=False)
    if p.returncode: git("pull", "--rebase", "origin", "main", check=False)
    if os.path.exists(os.path.join(REPO, "jobs", "STOP")): print("STOP file present: exiting"); return None
    publish_orphans()
    push_pending()                      # retry commits whose earlier push failed
    done = []
    for n in pending():
        print(f"[worker] running {n}", flush=True)
        rc = run_job(n); ok = publish(n[:-3]); print(f"[worker] {n} rc={rc} pushed={ok}", flush=True); done.append(n)
    return done


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--once", action="store_true"); ap.add_argument("--interval", type=int, default=30)
    a = ap.parse_args()
    tk = load_token()
    if tk: os.environ["GH_TOKEN"] = tk
    if not os.environ.get("GH_TOKEN") and not a.once and sys.stdin.isatty():
        os.environ["GH_TOKEN"] = clean_token(getpass.getpass("GitHub fine-grained token (this repo only, Contents RW; input hidden): "))
    d = describe_token(os.environ.get("GH_TOKEN", ""))
    print(f"[worker] repo={REPO} interval={a.interval}s token: len={d['length']} prefix={d['prefix']} -> {d['hint']}", flush=True)
    ok, msg = auth_check(); print(f"[worker] GitHub auth check: {'OK' if ok else 'REJECTED: ' + msg}", flush=True)
    while True:
        try:
            r = cycle()
            if r is None: sys.exit(3)                      # STOP file: tell the supervisor not to restart
            if a.once: break
        except KeyboardInterrupt: break
        except Exception as e: print("[worker] error:", e, flush=True)
        time.sleep(a.interval)
