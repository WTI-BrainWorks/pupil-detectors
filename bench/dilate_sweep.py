"""Sweep dark-mask dilate / spec erode configs vs gold + speed."""
import os, sys, subprocess, statistics
PY = r"c:/Users/adf44/source/python/pupil-detectors/.venv-3dvog/Scripts/python.exe"
WORKER = r'''
import os,sys,time; os.add_dll_directory(r"c:/tools/opencv/build/x64/vc16/bin"); sys.path.insert(0,"bench")
import numpy as np; from bench import load_frames
from pupil_detectors import Detector2D
seg=sys.argv[1]; gold=sys.argv[2]
fr=load_frames("bench/%s.raw"%seg); g=np.load(gold)
fids=g["fids"]; gv=g["valid"]; gcx=g["cx"]; gcy=g["cy"]; gd=np.maximum(g["MA"],g["ma"]); fset={int(f):k for k,f in enumerate(fids)}
d=Detector2D()
for i in range(40): d.detect(fr[i])
ts=np.empty(len(fr)); conf=np.zeros(len(fr)); cx=np.full(len(fr),np.nan); cy=np.full(len(fr),np.nan); dia=np.full(len(fr),np.nan)
for i in range(len(fr)):
    a=time.perf_counter(); r=d.detect(fr[i]); ts[i]=time.perf_counter()-a
    conf[i]=r["confidence"]; cx[i],cy[i]=r["ellipse"]["center"]; dia[i]=r["diameter"]
det=conf>0.5; both=[i for i in range(len(fr)) if det[i] and (i in fset) and gv[fset[i]]]
dist=np.array([np.hypot(cx[i]-gcx[fset[i]],cy[i]-gcy[fset[i]]) for i in both]); dd=np.array([abs(dia[i]-gd[fset[i]]) for i in both])
print("MED %.4f DET %d CMEAN %.4f CMAX %.4f DMEAN %.4f"%(np.median(ts*1000),det.sum(),dist.mean(),dist.max(),dd.mean()))
'''
configs = [
    ("dil7x2 ell, ero7 (base)", {}),
    ("dil7x1",                  {"PUPIL_DILATE_ITER": "1"}),
    ("dil5x2",                  {"PUPIL_DILATE_K": "5"}),
    ("dil5x1",                  {"PUPIL_DILATE_K": "5", "PUPIL_DILATE_ITER": "1"}),
    ("dil7x2 rect(sep)",        {"PUPIL_DILATE_RECT": "1"}),
    ("dil9x1 rect(sep)",        {"PUPIL_DILATE_K": "9", "PUPIL_DILATE_ITER": "1", "PUPIL_DILATE_RECT": "1"}),
    ("dil7x2 + ero5",           {"PUPIL_ERODE_K": "5"}),
]
def worker(seg, gold, env):
    e = dict(os.environ); e.update(env)
    o = subprocess.run([PY, "-c", WORKER, seg, gold], capture_output=True, text=True, env=e)
    toks = o.stdout.split()
    if "MED" not in toks: print("ERR", o.stderr[-400:]); sys.exit(1)
    return {toks[i]: float(toks[i + 1]) for i in range(0, len(toks), 2)}
for seg, gold in [("frames_2000", "bench/dv_gold.npz"), ("frames_blink", "bench/dv_gold_blink.npz")]:
    print(f"\n### {seg} (dilate/erode sweep; speed median of 4) ###")
    print(f"{'config':24s} {'ms':>7} {'spd':>6} {'det@0.5':>8} {'cmean':>7} {'cmax':>8} {'dmean':>7}")
    base = None
    for name, env in configs:
        meds = [worker(seg, gold, env) for _ in range(4)]
        ms = statistics.median([m["MED"] for m in meds]); r = meds[0]
        if base is None: base = ms
        print(f"{name:24s} {ms:7.4f} {base/ms:5.2f}x {int(r['DET']):8d} {r['CMEAN']:7.4f} {r['CMAX']:8.3f} {r['DMEAN']:7.4f}")
