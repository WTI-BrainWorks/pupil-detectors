"""Sweep MORPH_OPEN configs (PUPIL_OPEN_K / PUPIL_OPEN_RECT): speed + gold agreement.
Driver spawns one worker process per config (kernel is read once per process)."""
import os, sys, subprocess, statistics
import numpy as np

PY = r"c:/Users/adf44/source/python/pupil-detectors/.venv-3dvog/Scripts/python.exe"

WORKER = r'''
import os,sys,time; os.add_dll_directory(r"c:/tools/opencv/build/x64/vc16/bin"); sys.path.insert(0,"bench")
import numpy as np; from bench import load_frames
from pupil_detectors import Detector2D
seg=sys.argv[1]; gold=sys.argv[2]
fr=load_frames("bench/%s.raw"%seg); g=np.load(gold)
fids=g["fids"]; gv=g["valid"]; gcx=g["cx"]; gcy=g["cy"]; gd=np.maximum(g["MA"],g["ma"])
fset={int(f):k for k,f in enumerate(fids)}
d=Detector2D()
for i in range(40): d.detect(fr[i])
ts=np.empty(len(fr)); conf=np.zeros(len(fr)); cx=np.full(len(fr),np.nan); cy=np.full(len(fr),np.nan); dia=np.full(len(fr),np.nan)
for i in range(len(fr)):
    a=time.perf_counter(); r=d.detect(fr[i]); ts[i]=time.perf_counter()-a
    conf[i]=r["confidence"]; cx[i],cy[i]=r["ellipse"]["center"]; dia[i]=r["diameter"]
det=conf>0.5
both=[i for i in range(len(fr)) if det[i] and (i in fset) and gv[fset[i]]]
dist=np.array([np.hypot(cx[i]-gcx[fset[i]],cy[i]-gcy[fset[i]]) for i in both])
dd=np.array([abs(dia[i]-gd[fset[i]]) for i in both])
print("MED %.4f DET %d CMEAN %.4f CMAX %.4f DMEAN %.4f"%(np.median(ts*1000),det.sum(),dist.mean(),dist.max(),dd.mean()))
'''

configs = [
    ("9 ellipse (baseline)", {"PUPIL_OPEN_K": "9"}),
    ("7 ellipse", {"PUPIL_OPEN_K": "7"}),
    ("5 ellipse", {"PUPIL_OPEN_K": "5"}),
    ("9 rect(sep)", {"PUPIL_OPEN_K": "9", "PUPIL_OPEN_RECT": "1"}),
    ("7 rect(sep)", {"PUPIL_OPEN_K": "7", "PUPIL_OPEN_RECT": "1"}),
    ("none", {"PUPIL_OPEN_K": "0"}),
]


def worker(seg, gold, env):
    e = dict(os.environ); e.update(env)
    o = subprocess.run([PY, "-c", WORKER, seg, gold], capture_output=True, text=True, env=e)
    d = {}
    for tok, val in zip(o.stdout.split()[0::2], o.stdout.split()[1::2]):
        d[tok] = float(val)
    if "MED" not in d:
        print("ERR", o.stderr[-400:]); sys.exit(1)
    return d


for seg, gold in [("frames_2000", "bench/dv_gold.npz"), ("frames_blink", "bench/dv_gold_blink.npz")]:
    print(f"\n### {seg} (MORPH_OPEN sweep; speed = median of 4 runs) ###")
    print(f"{'config':22s} {'ms':>7} {'speedup':>8} {'det@0.5':>8} {'center_mean':>11} {'center_max':>10} {'diam_mean':>9}")
    base_ms = None
    for name, env in configs:
        meds = [worker(seg, gold, env) for _ in range(4)]
        ms = statistics.median([m["MED"] for m in meds]); r = meds[0]
        if base_ms is None: base_ms = ms
        print(f"{name:22s} {ms:7.4f} {base_ms/ms:7.2f}x {int(r['DET']):8d} {r['CMEAN']:11.4f} {r['CMAX']:10.3f} {r['DMEAN']:9.4f}")
