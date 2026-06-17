"""Per-frame runtime distribution: optimized baseline vs full algorithmic branch.
Both come from the current wheel via env toggles (baseline = pre-excursion config).
Runs each config in its own subprocess (env is read once at import)."""
import os, sys, subprocess
import numpy as np

PY = r"c:/Users/adf44/source/python/pupil-detectors/.venv-3dvog/Scripts/python.exe"
SEG = sys.argv[1] if len(sys.argv) > 1 else "frames_2000"

WORKER = r'''
import os,sys,time; os.add_dll_directory(r"c:/tools/opencv/build/x64/vc16/bin"); sys.path.insert(0,"bench")
import numpy as np; from bench import load_frames
from pupil_detectors import Detector2D
fr=load_frames("bench/%s.raw"); d=Detector2D()
for i in range(60): d.detect(fr[i])
ts=np.empty(len(fr))
for i in range(len(fr)):
    a=time.perf_counter(); d.detect(fr[i]); ts[i]=time.perf_counter()-a
np.save(sys.argv[1], ts*1000.0)
''' % SEG

# baseline-equivalent = every excursion toggled off
BASE = {"PUPIL_BLUR": "median", "PUPIL_OPEN_K": "9", "PUPIL_DILATE_RECT": "0", "PUPIL_PRIOR_ROI": "0"}
FULL = {}  # defaults = full branch


def collect(env, out):
    e = dict(os.environ); e.update(env)
    subprocess.run([PY, "-c", WORKER, out], env=e, check=True)
    return np.load(out)


base = collect(BASE, "bench/_ts_base.npy")
full = collect(FULL, "bench/_ts_full.npy")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(9, 5))
hi = np.percentile(np.concatenate([base, full]), 99.5)
bins = np.linspace(0, hi, 80)
ax.hist(base, bins=bins, alpha=0.55, label=f"optimized baseline (median={np.median(base):.3f} ms)", color="#888")
ax.hist(full, bins=bins, alpha=0.55, label=f"algorithmic branch (median={np.median(full):.3f} ms)", color="#2a7")
for arr, c in [(base, "#555"), (full, "#185")]:
    ax.axvline(np.median(arr), color=c, ls="--", lw=1)
ax.axvline(1.0, color="r", ls=":", lw=1, label="1 ms")
ax.set_xlabel("per-frame detect time (ms)"); ax.set_ylabel("frame count")
ax.set_title(f"Per-frame runtime distribution  ({SEG}, {len(full)} frames, single-thread)\n"
             f"baseline mean {base.mean():.3f} / p99 {np.percentile(base,99):.3f}  ->  "
             f"branch mean {full.mean():.3f} / p99 {np.percentile(full,99):.3f} ms")
ax.legend()
fig.tight_layout()
out = f"bench/runtime_hist_{SEG}.png"
fig.savefig(out, dpi=110)
print(f"baseline: mean={base.mean():.3f} median={np.median(base):.3f} p99={np.percentile(base,99):.3f} >1ms={100*(base>1).mean():.1f}%")
print(f"branch:   mean={full.mean():.3f} median={np.median(full):.3f} p99={np.percentile(full,99):.3f} >1ms={100*(full>1).mean():.1f}%")
print(f"-> {out}")
