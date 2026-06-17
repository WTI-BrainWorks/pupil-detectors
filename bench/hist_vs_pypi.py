"""Overlay per-frame runtime distributions: current build vs the PyPI release.

Driver collects per-frame detect times from two venvs (subprocess each, since
they have different pupil_detectors installed) and plots overlaid histograms.

  python bench/hist_vs_pypi.py [frames_2000|frames_blink]
"""
import os, sys, subprocess
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CUR_PY = ROOT + "/.venv-3dvog/Scripts/python.exe"     # current consolidated branch
PYPI_PY = ROOT + "/.venv-pypi/Scripts/python.exe"     # pupil-detectors 2.0.2 from PyPI
SEG = sys.argv[1] if len(sys.argv) > 1 else "frames_2000"

WORKER = r'''
import os,sys,time; os.add_dll_directory(r"c:/tools/opencv/build/x64/vc16/bin"); sys.path.insert(0,"bench")
import numpy as np; from bench import load_frames
from pupil_detectors import Detector2D
fr=load_frames(r"bench/%s.raw"); d=Detector2D()
for i in range(60): d.detect(fr[i])
ts=np.empty(len(fr))
for i in range(len(fr)):
    a=time.perf_counter(); d.detect(fr[i]); ts[i]=(time.perf_counter()-a)*1000.0
np.save(sys.argv[1], ts)
''' % SEG


def collect(py, out):
    # strip venv vars so the (possibly different-Python) target venv stays isolated
    env = {k: v for k, v in os.environ.items()
           if k not in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "__PYVENV_LAUNCHER__")}
    r = subprocess.run([py, "-c", WORKER, out], cwd=ROOT, env=env, capture_output=True, text=True)
    if r.returncode:
        print("worker failed:", r.stderr[-500:]); sys.exit(1)
    return np.load(out)


cur = collect(CUR_PY, "bench/_ts_cur.npy")
pyp = collect(PYPI_PY, "bench/_ts_pypi.npy")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(9, 5))
hi = np.percentile(np.concatenate([cur, pyp]), 99)
bins = np.linspace(0, hi, 80)
ax.hist(pyp, bins=bins, alpha=0.55, color="#c66", label=f"PyPI 2.0.2 (median {np.median(pyp):.3f} ms)")
ax.hist(cur, bins=bins, alpha=0.6, color="#2a7", label=f"optimized branch (median {np.median(cur):.3f} ms)")
ax.axvline(np.median(pyp), color="#a33", ls="--", lw=1)
ax.axvline(np.median(cur), color="#185", ls="--", lw=1)
ax.axvline(1.0, color="k", ls=":", lw=1, label="1 ms")
ax.set_xlabel("per-frame detect time (ms)"); ax.set_ylabel("frame count")
ax.set_title(f"pupil_detectors per-frame runtime: PyPI 2.0.2 vs optimized branch\n"
             f"({SEG}, {len(cur)} frames)  "
             f"PyPI mean {pyp.mean():.3f} / p99 {np.percentile(pyp,99):.2f}  ->  "
             f"branch mean {cur.mean():.3f} / p99 {np.percentile(cur,99):.2f} ms  "
             f"({pyp.mean()/cur.mean():.1f}x)")
ax.legend(); fig.tight_layout()
out = f"bench/hist_vs_pypi_{SEG}.png"
fig.savefig(out, dpi=110)
print(f"PyPI : mean={pyp.mean():.3f} median={np.median(pyp):.3f} p99={np.percentile(pyp,99):.3f} >1ms={100*(pyp>1).mean():.1f}%")
print(f"branch: mean={cur.mean():.3f} median={np.median(cur):.3f} p99={np.percentile(cur,99):.3f} >1ms={100*(cur>1).mean():.1f}%")
print(f"mean speedup {pyp.mean()/cur.mean():.2f}x  median {np.median(pyp)/np.median(cur):.2f}x  -> {out}")
