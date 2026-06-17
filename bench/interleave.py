"""Interleaved timing: alternately run reference and candidate processes many
times to control for machine drift. Reports median-of-run-medians per build.
"""
import subprocess, sys, re, statistics, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# REF / CAND venv names can be overridden via env (default: reference vs optimized)
REF_VENV = os.environ.get("REF_VENV", ".venv-bench")
CAND_VENV = os.environ.get("CAND_VENV", ".venv")
REF_PY = os.path.join(ROOT, REF_VENV, "Scripts", "python.exe")
CAND_PY = os.path.join(ROOT, CAND_VENV, "Scripts", "python.exe")
N_ROUNDS = int(sys.argv[1]) if len(sys.argv) > 1 else 8

# A short timing-only run script fed via stdin.
SNIPPET = r'''
import os, time, numpy as np
for _d in [r"c:/tools/opencv/build/x64/vc16/bin",
           r"{root}/.deps/opencv5/opencv/build/x64/vc16/bin"]:
    if os.path.isdir(_d):
        os.add_dll_directory(_d)
import sys; sys.path.insert(0, r"{root}/bench")
from bench import load_frames
from pupil_detectors import Detector2D
fr = load_frames(r"{root}/bench/frames_2000.raw")
d = Detector2D()
for i in range(40): d.detect(fr[i])
ts = np.empty(len(fr))
for i in range(len(fr)):
    a = time.perf_counter(); d.detect(fr[i]); ts[i] = time.perf_counter()-a
ms = ts*1000
print("STATS %.4f %.4f %.4f %.4f %.4f %.4f" % (
    np.median(ms), ms.mean(), np.percentile(ms,90), np.percentile(ms,95),
    np.percentile(ms,99), 100.0*(ms>1).mean()))
'''.format(root=ROOT.replace("\\", "/"))

KEYS = ["med", "mean", "p90", "p95", "p99", "pct_gt1"]


def _env(spec):
    e = dict(os.environ)
    for kv in spec.split(",") if spec else []:
        if "=" in kv:
            k, v = kv.split("=", 1); e[k] = v
    return e


REF_ENV = _env(os.environ.get("REF_ENV", ""))
CAND_ENV = _env(os.environ.get("CAND_ENV", ""))


def one(py, env):
    out = subprocess.run([py, "-c", SNIPPET], capture_output=True, text=True, env=env)
    m = re.search(r"STATS " + " ".join([r"([\d.]+)"]*6), out.stdout)
    if not m:
        print("ERR:", out.stdout, out.stderr[-800:]); sys.exit(1)
    return dict(zip(KEYS, map(float, m.groups())))


ref = {k: [] for k in KEYS}
cand = {k: [] for k in KEYS}
for r in range(N_ROUNDS):
    if r % 2 == 0:
        a = one(REF_PY, REF_ENV); b = one(CAND_PY, CAND_ENV)
    else:
        b = one(CAND_PY, CAND_ENV); a = one(REF_PY, REF_ENV)
    for k in KEYS:
        ref[k].append(a[k]); cand[k].append(b[k])
    print(f"round {r}: ref med={a['med']:.3f} p95={a['p95']:.3f} p99={a['p99']:.3f} >1ms={a['pct_gt1']:.0f}% "
          f"| cand med={b['med']:.3f} p95={b['p95']:.3f} p99={b['p99']:.3f} >1ms={b['pct_gt1']:.0f}%")

print("\n=== summary over", N_ROUNDS, "rounds (median of per-run values) ===")
print(f"{'metric':>8}  {'reference':>10}  {'candidate':>10}  {'speedup':>8}")
for k in KEYS:
    rv = statistics.median(ref[k]); cv = statistics.median(cand[k])
    sp = f"{rv/cv:.3f}x" if k != "pct_gt1" and cv else "-"
    print(f"{k:>8}  {rv:>10.4f}  {cv:>10.4f}  {sp:>8}")
