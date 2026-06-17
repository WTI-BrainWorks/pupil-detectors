"""Sweep PUPIL_MAX_EVALS: tail latency (p99/max) + gold agreement."""
import os, sys, subprocess
import numpy as np
PY = r"c:/Users/adf44/source/python/pupil-detectors/.venv-3dvog/Scripts/python.exe"
WORKER = r'''
import os,sys,time; os.add_dll_directory(r"c:/tools/opencv/build/x64/vc16/bin"); sys.path.insert(0,"bench")
import numpy as np; from bench import load_frames
from pupil_detectors import Detector2D
seg=sys.argv[1]; gold=sys.argv[2]
fr=load_frames("bench/%s.raw"%seg); g=np.load(gold)
fids=g["fids"]; gv=g["valid"]; gcx=g["cx"]; gcy=g["cy"]; fset={int(f):k for k,f in enumerate(fids)}
d=Detector2D()
for i in range(60): d.detect(fr[i])
best=None
for rep in range(3):
    ts=np.empty(len(fr)); conf=np.zeros(len(fr)); cx=np.full(len(fr),np.nan); cy=np.full(len(fr),np.nan)
    for i in range(len(fr)):
        a=time.perf_counter(); r=d.detect(fr[i]); ts[i]=time.perf_counter()-a
        conf[i]=r["confidence"]; cx[i],cy[i]=r["ellipse"]["center"]
    ms=ts*1000.0
    if best is None or np.median(ms)<best[0]: best=(np.median(ms),np.percentile(ms,99),ms.max())
det=conf>0.5
both=[i for i in range(len(fr)) if det[i] and (i in fset) and gv[fset[i]]]
dist=np.array([np.hypot(cx[i]-gcx[fset[i]],cy[i]-gcy[fset[i]]) for i in both])
print("MED %.4f P99 %.4f MAX %.4f DET %d CMEAN %.4f"%(best[0],best[1],best[2],det.sum(),dist.mean()))
'''
def worker(seg,gold,me):
    e=dict(os.environ); e["PUPIL_MAX_EVALS"]=str(me)
    o=subprocess.run([PY,"-c",WORKER,seg,gold],capture_output=True,text=True,env=e)
    t=o.stdout.split()
    if "MED" not in t: print("ERR",o.stderr[-300:]); sys.exit(1)
    return {t[i]:float(t[i+1]) for i in range(0,len(t),2)}
for seg,gold in [("frames_2000","bench/dv_gold.npz"),("frames_blink","bench/dv_gold_blink.npz")]:
    print(f"\n### {seg}: max_evals sweep (tail + gold) ###")
    print(f"{'max_evals':>9} {'median_ms':>10} {'p99_ms':>8} {'max_ms':>8} {'det@0.5':>8} {'cmean':>7}")
    for me in (1000,500,300,200,100,50):
        r=worker(seg,gold,me)
        print(f"{me:>9} {r['MED']:>10.4f} {r['P99']:>8.4f} {r['MAX']:>8.4f} {int(r['DET']):>8d} {r['CMEAN']:>7.4f}")
