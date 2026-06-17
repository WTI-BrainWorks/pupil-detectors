"""Plot the per-frame detect-time distribution for a segment (current build).

  python bench/runtime_hist.py [frames_2000|frames_blink]

For a before/after comparison use bench/interleave.py against a baseline wheel.
"""
import os, sys, time
os.add_dll_directory(r"c:/tools/opencv/build/x64/vc16/bin")
sys.path.insert(0, "bench")
import numpy as np
from bench import load_frames
from pupil_detectors import Detector2D

SEG = sys.argv[1] if len(sys.argv) > 1 else "frames_2000"
fr = load_frames(f"bench/{SEG}.raw")
d = Detector2D()
for i in range(60):
    d.detect(fr[i])
ts = np.empty(len(fr))
for i in range(len(fr)):
    a = time.perf_counter(); d.detect(fr[i]); ts[i] = (time.perf_counter() - a) * 1000.0

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(9, 5))
hi = np.percentile(ts, 99.7)
ax.hist(ts, bins=np.linspace(0, hi, 80), color="#2a7", alpha=0.8)
ax.axvline(np.median(ts), color="#185", ls="--", lw=1, label=f"median {np.median(ts):.3f} ms")
ax.axvline(1.0, color="r", ls=":", lw=1, label="1 ms")
ax.set_xlabel("per-frame detect time (ms)"); ax.set_ylabel("frame count")
ax.set_title(f"Per-frame runtime ({SEG}, {len(ts)} frames, single-thread)\n"
             f"mean {ts.mean():.3f}  median {np.median(ts):.3f}  p99 {np.percentile(ts,99):.3f}  "
             f">1ms {100*(ts>1).mean():.1f}%")
ax.legend(); fig.tight_layout()
out = f"bench/runtime_hist_{SEG}.png"
fig.savefig(out, dpi=110)
print(f"{SEG}: mean={ts.mean():.3f} median={np.median(ts):.3f} p99={np.percentile(ts,99):.3f} "
      f"max={ts.max():.3f} >1ms={100*(ts>1).mean():.1f}%  -> {out}")
