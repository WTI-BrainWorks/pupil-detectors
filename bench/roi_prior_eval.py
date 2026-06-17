"""Prototype + evaluate prior-seeded ROI (skip coarse detection on tracked frames).

Strategy: if the previous frame was a confident detection, seed a tight ROI from
its center/diameter (bypasses coarse detection). If detection in that ROI comes
back weak, fall back to a full-frame detect (coarse relocates the pupil).

Compares against the cached 3DeepVOG gold + baseline (coarse every frame).
"""
import os, sys, time
os.add_dll_directory(r"c:/tools/opencv/build/x64/vc16/bin")
sys.path.insert(0, "bench")
import numpy as np
from bench import load_frames
from pupil_detectors import Detector2D, Roi

LOST = 0.3   # confidence below which we consider the tight-ROI detect a miss and retry
KEEP = 0.6   # confidence above which we trust the result to seed next frame's ROI


def clampbox(cx, cy, m):
    x0, y0 = max(0, int(cx - m)), max(0, int(cy - m))
    x1, y1 = min(640, int(cx + m)), min(480, int(cy + m))
    return Roi(x0, y0, x1, y1)


def run(seg, margin_mult, fallback=True):
    fr = load_frames(f"bench/{seg}.raw")
    d = Detector2D()
    for i in range(40):
        d.detect(fr[i])
    n = len(fr)
    conf = np.zeros(n); cx = np.full(n, np.nan); cy = np.full(n, np.nan); dia = np.full(n, np.nan)
    times = np.empty(n); retries = 0; coarse_used = 0
    prev = None
    for i in range(n):
        t0 = time.perf_counter()
        if prev is not None:
            pcx, pcy, pdia = prev
            roi = clampbox(pcx, pcy, max(55.0, pdia * margin_mult))
            r = d.detect(fr[i], roi=roi)
            if fallback and r["confidence"] < LOST:
                r = d.detect(fr[i])  # relocate via coarse
                coarse_used += 1; retries += 1
        else:
            r = d.detect(fr[i]); coarse_used += 1
        times[i] = time.perf_counter() - t0
        conf[i] = r["confidence"]; dia[i] = r["diameter"]
        cx[i], cy[i] = r["ellipse"]["center"]
        prev = (cx[i], cy[i], dia[i]) if r["confidence"] > KEEP else None
    return dict(conf=conf, cx=cx, cy=cy, dia=dia, times=times, retries=retries, coarse_used=coarse_used)


def baseline(seg):
    fr = load_frames(f"bench/{seg}.raw"); d = Detector2D()
    for i in range(40): d.detect(fr[i])
    n = len(fr); conf = np.zeros(n); cx = np.full(n, np.nan); cy = np.full(n, np.nan); dia = np.full(n, np.nan); times = np.empty(n)
    for i in range(n):
        t0 = time.perf_counter(); r = d.detect(fr[i]); times[i] = time.perf_counter() - t0
        conf[i] = r["confidence"]; dia[i] = r["diameter"]; cx[i], cy[i] = r["ellipse"]["center"]
    return dict(conf=conf, cx=cx, cy=cy, dia=dia, times=times, retries=0, coarse_used=n)


def agree(res, gold, thr=0.5):
    g = np.load(gold); fids = g["fids"]; gv = g["valid"]; gcx = g["cx"]; gcy = g["cy"]; gd = np.maximum(g["MA"], g["ma"])
    det = res["conf"] > thr
    idx = [int(f) for f in fids]
    both = np.array([det[i] and gv[k] for k, i in enumerate(idx)])
    dist = np.array([np.hypot(res["cx"][i] - gcx[k], res["cy"][i] - gcy[k]) for k, i in enumerate(idx) if both[k]])
    dd = np.array([abs(res["dia"][i] - gd[k]) for k, i in enumerate(idx) if both[k]])
    return det.sum(), both.sum(), dist, dd


for seg, gold in [("frames_2000", "bench/dv_gold.npz"), ("frames_blink", "bench/dv_gold_blink.npz")]:
    print(f"\n##### {seg} #####")
    base = baseline(seg)
    bd, bb, bdist, bddiam = agree(base, gold)
    print(f"baseline (coarse every frame): time mean={base['times'].mean()*1e3:.3f} median={np.median(base['times'])*1e3:.3f}ms | "
          f"det@0.5={bd} | center mean={bdist.mean():.3f} max={bdist.max():.3f} | diam {bddiam.mean():.3f}")
    for mm in (1.0, 0.8):
        res = run(seg, mm)
        rd, rb, rdist, rddiam = agree(res, gold)
        print(f"prior-ROI mult={mm}: time mean={res['times'].mean()*1e3:.3f} median={np.median(res['times'])*1e3:.3f}ms "
              f"(speedup {base['times'].mean()/res['times'].mean():.2f}x) | det@0.5={rd} | "
              f"center mean={rdist.mean():.3f} max={rdist.max():.3f} | diam {rddiam.mean():.3f} | "
              f"coarse_used={res['coarse_used']} retries={res['retries']}")
