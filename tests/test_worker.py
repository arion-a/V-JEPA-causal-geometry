import json, os, subprocess, sys

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))


def g(cwd, *a):
    return subprocess.run(["git", "-C", cwd, "-c", "user.name=t", "-c", "user.email=t@t", *a], capture_output=True, text=True, check=True).stdout


def setup(tmp):
    bare = f"{tmp}/remote.git"; subprocess.run(["git", "init", "-q", "--bare", "-b", "main", bare], check=True)
    pod, me = f"{tmp}/pod", f"{tmp}/me"
    for d in (pod, me): subprocess.run(["git", "clone", "-q", bare, d], check=True, capture_output=True)
    os.makedirs(f"{me}/scripts"); os.makedirs(f"{me}/jobs")
    open(f"{me}/scripts/pod_worker.py", "w").write(open(f"{ROOT}/scripts/pod_worker.py").read())
    open(f"{me}/.gitignore", "w").write(open(f"{ROOT}/.gitignore").read())     # use the REAL ignore rules
    g(me, "checkout", "-q", "-b", "main"); g(me, "add", "-A"); g(me, "commit", "-qm", "init"); g(me, "push", "-q", "origin", "main")
    g(pod, "pull", "-q", "origin", "main"); g(pod, "branch", "-q", "--set-upstream-to=origin/main", "main")
    return pod, me


def worker(pod, *args, **env):
    return subprocess.run([sys.executable, f"{pod}/scripts/pod_worker.py", "--once", *args], capture_output=True, text=True,
                          env={**os.environ, "CG_ROOT": f"{pod}/vol", **env})


def test_runs_pending_job_once_and_publishes(tmp_path):
    pod, me = setup(str(tmp_path))
    open(f"{me}/jobs/001_hello.sh", "w").write('echo hi; echo data > "$JOB_OUT/out.txt"; test -z "$GH_TOKEN" && echo notoken\n')
    g(me, "add", "-A"); g(me, "commit", "-qm", "job"); g(me, "push", "-q", "origin", "main")
    r = worker(pod, GH_TOKEN="secret-token-123"); assert "rc=0 pushed=True" in r.stdout, r.stdout + r.stderr
    g(me, "pull", "-q", "origin", "main")
    d = f"{me}/results/jobs/001_hello"
    st = json.load(open(f"{d}/status.json")); assert st["rc"] == 0 and len(st["job_sha256"]) == 64
    assert open(f"{d}/out.txt").read() == "data\n"
    log = open(f"{d}/log.txt").read(); assert "hi" in log and "notoken" in log and "secret-token-123" not in log   # token not exposed to jobs
    assert "secret-token-123" not in open(f"{pod}/.git/config").read()
    r2 = worker(pod); assert "running" not in r2.stdout                                                         # idempotent: not re-run


def test_failure_timeout_large_file_and_stop(tmp_path):
    pod, me = setup(str(tmp_path))
    open(f"{me}/jobs/001_fail.sh", "w").write("exit 3\n")
    open(f"{me}/jobs/002_slow.sh", "w").write("# timeout: 1\nsleep 5\n")
    open(f"{me}/jobs/003_big.sh", "w").write('head -c 1500000 /dev/zero > "$JOB_OUT/big.bin"\n')
    g(me, "add", "-A"); g(me, "commit", "-qm", "jobs"); g(me, "push", "-q", "origin", "main")
    worker(pod); g(me, "pull", "-q", "origin", "main")
    rc = lambda n: json.load(open(f"{me}/results/jobs/{n}/status.json"))["rc"]
    assert rc("001_fail") == 3 and rc("002_slow") == -9
    assert not os.path.exists(f"{me}/results/jobs/003_big/big.bin") and os.path.exists(f"{me}/results/jobs/003_big/big.bin.TOO_LARGE")
    open(f"{me}/jobs/004_after_stop.sh", "w").write("echo x\n"); open(f"{me}/jobs/STOP", "w").write("")
    g(me, "add", "-A"); g(me, "commit", "-qm", "stop"); g(me, "push", "-q", "origin", "main")
    assert "STOP" in worker(pod).stdout
    g(me, "pull", "-q", "origin", "main"); assert not os.path.exists(f"{me}/results/jobs/004_after_stop")


def test_unpublished_results_are_published_not_rerun(tmp_path):
    pod, me = setup(str(tmp_path))
    open(f"{me}/jobs/001_once.sh", "w").write('echo run >> "$CG_ROOT/count.txt"; echo ok > "$JOB_OUT/o.txt"\n')
    g(me, "add", "-A"); g(me, "commit", "-qm", "job"); g(me, "push", "-q", "origin", "main")
    g(pod, "pull", "-q", "origin", "main")
    os.makedirs(f"{pod}/vol")
    os.makedirs(f"{pod}/results/jobs/001_once"); open(f"{pod}/results/jobs/001_once/status.json", "w").write(json.dumps(dict(rc=0)))
    open(f"{pod}/results/jobs/001_once/o.txt", "w").write("ok\n")      # ran earlier, never committed
    r = worker(pod); assert "publishing unpublished" in r.stdout, r.stdout + r.stderr
    g(me, "pull", "-q", "origin", "main"); assert os.path.exists(f"{me}/results/jobs/001_once/o.txt")
    assert not os.path.exists(f"{pod}/vol/count.txt")                  # job was not re-run


def test_failed_push_is_retried_next_cycle(tmp_path):
    pod, me = setup(str(tmp_path))
    open(f"{me}/jobs/001_a.sh", "w").write('echo a > "$JOB_OUT/a.txt"\n')
    g(me, "add", "-A"); g(me, "commit", "-qm", "job"); g(me, "push", "-q", "origin", "main")
    bare = f"{tmp_path}/remote.git"; hook = f"{bare}/hooks/pre-receive"
    open(hook, "w").write("#!/bin/sh\nexit 1\n"); os.chmod(hook, 0o755)                    # remote rejects pushes
    r = worker(pod); assert "PUSH FAILED" in r.stdout and "1 commit" in r.stdout, r.stdout + r.stderr
    os.remove(hook)                                                                           # remote recovers
    worker(pod); g(me, "pull", "-q", "origin", "main")
    assert os.path.exists(f"{me}/results/jobs/001_a/a.txt")
