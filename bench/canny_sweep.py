"""Sweep Canny aperture/threshold (runtime properties, no rebuild): speed + gold."""
import os, sys, time
os.add_dll_directory(r"c:/tools/opencv/build/x64/vc16/bin")
sys.path.insert(0, "bench")
import numpy as np
from bench import load_frames
from pupil_detectors import Detector2D


def agree(seg, gold, props):
    fr = load_frames(f"bench/{seg}.raw"); g = np.load(gold)
    fids = g["fids"]; gv = g["valid"]; gcx = g["cx"]; gcy = g["cy"]; gd = np.maximum(g["MA"], g["ma"])
    fset = {int(f): k for k, f in enumerate(fids)}
    d = Detector2D(props)
    for i in range(40): d.detect(fr[i])
    ts = np.empty(len(fr)); conf = np.zeros(len(fr)); cx = np.full(len(fr), np.nan); cy = np.full(len(fr), np.nan); dia = np.full(len(fr), np.nan)
    for i in range(len(fr)):
        a = time.perf_counter(); r = d.detect(fr[i]); ts[i] = time.perf_counter() - a
        conf[i] = r["confidence"]; cx[i], cy[i] = r["ellipse"]["center"]; dia[i] = r["diameter"]
    det = conf > 0.5
    both = [i for i in range(len(fr)) if det[i] and (i in fset) and gv[fset[i]]]
    dist = np.array([np.hypot(cx[i] - gcx[fset[i]], cy[i] - gcy[fset[i]]) for i in both])
    if dist.size == 0:
        return np.median(ts * 1000), int(det.sum()), float("nan"), float("nan")
    return np.median(ts * 1000), int(det.sum()), dist.mean(), dist.max()


configs = [
    ("aperture 5 (base)", {}),
    ("aperture 3", {"canny_aperture": 3}),
    ("aperture 7", {"canny_aperture": 7}),
    ("aperture 3, thresh 100", {"canny_aperture": 3, "canny_treshold": 100}),
    ("aperture 3, thresh 60", {"canny_aperture": 3, "canny_treshold": 60}),
]
for seg, gold in [("frames_2000", "bench/dv_gold.npz"), ("frames_blink", "bench/dv_gold_blink.npz")]:
    print(f"\n### {seg} (Canny sweep) ###")
    print(f"{'config':24s} {'ms':>7} {'det@0.5':>8} {'center_mean':>11} {'center_max':>10}")
    base = None
    for name, p in configs:
        # median of 3 timing runs
        runs = [agree(seg, gold, p) for _ in range(3)]
        ms = float(np.median([r[0] for r in runs])); r = runs[0]
        if base is None: base = ms
        print(f"{name:24s} {ms:7.4f} {r[1]:8d} {r[2]:11.4f} {r[3]:10.3f}  ({base/ms:.2f}x)")
