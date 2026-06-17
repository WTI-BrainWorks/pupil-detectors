"""Interleaved timing: alternately run reference and candidate processes many
times to control for machine drift. Reports median-of-run-medians per build.
"""
import subprocess, sys, re, statistics, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF_PY = os.path.join(ROOT, ".venv", "Scripts", "python.exe")
CAND_PY = os.path.join(ROOT, ".venv-bench", "Scripts", "python.exe")
N_ROUNDS = int(sys.argv[1]) if len(sys.argv) > 1 else 8

# A short timing-only run script fed via stdin.
SNIPPET = r'''
import os, time, numpy as np
os.add_dll_directory(r"c:/tools/opencv/build/x64/vc16/bin")
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
print("MED %.4f MEAN %.4f" % (np.median(ms), ms.mean()))
'''.format(root=ROOT.replace("\\", "/"))


def one(py):
    out = subprocess.run([py, "-c", SNIPPET], capture_output=True, text=True)
    m = re.search(r"MED ([\d.]+) MEAN ([\d.]+)", out.stdout)
    if not m:
        print("ERR:", out.stdout, out.stderr[-500:]); sys.exit(1)
    return float(m.group(1)), float(m.group(2))


ref_med, ref_mean, cand_med, cand_mean = [], [], [], []
for r in range(N_ROUNDS):
    # alternate order each round to avoid systematic bias
    if r % 2 == 0:
        a = one(REF_PY); b = one(CAND_PY)
    else:
        b = one(CAND_PY); a = one(REF_PY)
    ref_med.append(a[0]); ref_mean.append(a[1])
    cand_med.append(b[0]); cand_mean.append(b[1])
    print(f"round {r}: ref med={a[0]:.4f} mean={a[1]:.4f} | cand med={b[0]:.4f} mean={b[1]:.4f}")

rm, cm = statistics.median(ref_med), statistics.median(cand_med)
rmn, cmn = statistics.median(ref_mean), statistics.median(cand_mean)
print("\n=== summary over", N_ROUNDS, "rounds (median of per-run values) ===")
print(f"reference  median/frame: {rm:.4f} ms   mean/frame: {rmn:.4f} ms")
print(f"candidate  median/frame: {cm:.4f} ms   mean/frame: {cmn:.4f} ms")
print(f"speedup (median-of-medians): {rm/cm:.3f}x")
print(f"speedup (median-of-means):   {rmn/cmn:.3f}x")
