"""Evaluate the dark-blob fallback: miss-recovery + false positives vs gold + speed.
Runs the detector with use_blob_fallback 0 and 1 over each segment and compares
both to the cached 3DeepVOG gold."""
import os, sys, time
os.add_dll_directory(r"c:/tools/opencv/build/x64/vc16/bin")
sys.path.insert(0, "bench")
import numpy as np
from bench import load_frames
from pupil_detectors import Detector2D

THR = 0.5


def run(seg, fb):
    fr = load_frames(f"bench/{seg}.raw")
    d = Detector2D({"use_blob_fallback": fb})
    n = len(fr)
    conf = np.zeros(n); cx = np.full(n, np.nan); cy = np.full(n, np.nan); dia = np.full(n, np.nan); ts = np.empty(n)
    for i in range(n):
        a = time.perf_counter(); r = d.detect(fr[i]); ts[i] = (time.perf_counter() - a) * 1000
        conf[i] = r["confidence"]; cx[i], cy[i] = r["ellipse"]["center"]; dia[i] = r["diameter"]
    return conf, cx, cy, dia, ts


for seg, gold in [("frames_2000", "bench/dv_gold.npz"), ("frames_blink", "bench/dv_gold_blink.npz")]:
    g = np.load(gold); fids = g["fids"]; gv = g["valid"]; gcx = g["cx"]; gcy = g["cy"]; gd = np.maximum(g["MA"], g["ma"])
    fset = {int(f): k for k, f in enumerate(fids)}
    off = run(seg, 0); on = run(seg, 1)
    print(f"\n### {seg} ###")
    for tag, (conf, cx, cy, dia, ts) in (("fallback OFF", off), ("fallback ON", on)):
        det = conf > THR
        both = [i for i in range(len(conf)) if det[i] and (i in fset) and gv[fset[i]]]
        dist = np.array([np.hypot(cx[i] - gcx[fset[i]], cy[i] - gcy[fset[i]]) for i in both])
        dd = np.array([abs(dia[i] - gd[fset[i]]) for i in both])
        print(f"  {tag:12s}: det@0.5={det.sum():4d}  center mean={dist.mean():.3f} p95={np.percentile(dist,95):.3f} "
              f"max={dist.max():.2f}  diam={dd.mean():.3f}  >3px={int((dist>3).sum())}  median_ms={np.median(ts):.3f}")
    # recovery analysis: gold-valid frames the OFF build missed (conf<=THR) that ON now detects
    offc, onc = off[0], on[0]
    gvi = [int(f) for k, f in enumerate(fids) if gv[k]]
    missed_off = [i for i in gvi if offc[i] <= THR]
    rec = [i for i in missed_off if onc[i] > THR]
    # of recovered, how close to gold?
    recd = [np.hypot(on[1][i] - gcx[fset[i]], on[2][i] - gcy[fset[i]]) for i in rec]
    good_rec = sum(1 for x in recd if x <= 3)
    print(f"  gold-valid frames missed by OFF: {len(missed_off)};  recovered by ON: {len(rec)}  "
          f"(within 3px of gold: {good_rec}, off: {len(rec)-good_rec})")
    # false positives introduced: frames where ON detects but OFF didn't, AND ON is far from gold (or gold invalid)
    new_det = [i for i in range(len(onc)) if onc[i] > THR and offc[i] <= THR]
    fp = [i for i in new_det if (i in fset) and (not gv[fset[i]] or np.hypot(on[1][i]-gcx[fset[i]], on[2][i]-gcy[fset[i]]) > 5)]
    print(f"  new detections by ON: {len(new_det)};  of those suspect (gold invalid or >5px): {len(fp)}")
