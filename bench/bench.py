"""Benchmark + accuracy harness for pupil_detectors.Detector2D.

Loads pre-extracted grayscale frames (raw uint8, 640x480) and runs the 2D
detector over them, recording per-frame timing and detection results.

The frame clips are extracted from the shared sample video, which now lives at
the pupil-pkgs top level (../../sample_data/eye1.mp4, shared with the pye3d
benchmark); regenerate them with extract_frames.py.

Usage:
    python bench/bench.py run   --tag reference --frames bench/frames_2000.raw
    python bench/bench.py run   --tag candidate --frames bench/frames_2000.raw
    python bench/bench.py cmp   --ref bench/out_reference.npz --cand bench/out_candidate.npz
"""
import argparse
import os
import sys
import time

# OpenCV C++ runtime DLLs (local build links system OpenCV, not bundled).
# Overridable via PUPIL_OPENCV_BIN for builds linked against a different OpenCV.
_OPENCV_BIN = os.environ.get("PUPIL_OPENCV_BIN", r"c:/tools/opencv/build/x64/vc16/bin")
if os.path.isdir(_OPENCV_BIN):
    os.add_dll_directory(_OPENCV_BIN)

import numpy as np

W, H = 640, 480


def load_frames(path, max_frames=None):
    data = np.fromfile(path, dtype=np.uint8)
    n = data.size // (W * H)
    data = data[: n * W * H].reshape(n, H, W)
    if max_frames:
        data = data[:max_frames]
    return np.ascontiguousarray(data)


def run(args):
    from pupil_detectors import Detector2D
    import pupil_detectors

    frames = load_frames(args.frames, args.max_frames)
    n = len(frames)
    print(f"[{args.tag}] pupil_detectors {pupil_detectors.__version__}, {n} frames", flush=True)

    det = Detector2D()

    # result arrays
    cx = np.empty(n); cy = np.empty(n)
    ax0 = np.empty(n); ax1 = np.empty(n)
    ang = np.empty(n); conf = np.empty(n); diam = np.empty(n)
    times = np.empty(n)

    # warmup (jit caches, allocations)
    for i in range(min(20, n)):
        det.detect(frames[i])

    t_all0 = time.perf_counter()
    for i in range(n):
        f = frames[i]
        t0 = time.perf_counter()
        r = det.detect(f)
        times[i] = time.perf_counter() - t0
        e = r["ellipse"]
        cx[i], cy[i] = e["center"]
        ax0[i], ax1[i] = e["axes"]
        ang[i] = e["angle"]
        conf[i] = r["confidence"]
        diam[i] = r["diameter"]
    t_all = time.perf_counter() - t_all0

    out = args.out or f"bench/out_{args.tag}.npz"
    np.savez(out, cx=cx, cy=cy, ax0=ax0, ax1=ax1, ang=ang, conf=conf, diam=diam, times=times)
    ms = times * 1000.0
    print(f"[{args.tag}] total {t_all:.3f}s  fps={n/t_all:.1f}")
    print(f"[{args.tag}] per-frame ms: mean={ms.mean():.3f} median={np.median(ms):.3f} "
          f"p95={np.percentile(ms,95):.3f} min={ms.min():.3f} max={ms.max():.3f}")
    print(f"[{args.tag}] mean confidence={conf.mean():.4f}  detections(conf>0)={int((conf>0).sum())}/{n}")
    print(f"[{args.tag}] saved -> {out}")


def cmp(args):
    a = np.load(args.ref)
    b = np.load(args.cand)
    n = len(a["conf"])
    print(f"frames={n}")
    # timing
    ta, tb = a["times"] * 1000, b["times"] * 1000
    print(f"ref   ms: mean={ta.mean():.3f} median={np.median(ta):.3f} p95={np.percentile(ta,95):.3f}")
    print(f"cand  ms: mean={tb.mean():.3f} median={np.median(tb):.3f} p95={np.percentile(tb,95):.3f}")
    print(f"speedup (mean)  = {ta.mean()/tb.mean():.3f}x")
    print(f"speedup (median)= {np.median(ta)/np.median(tb):.3f}x")
    print("--- accuracy vs reference ---")
    dconf = np.abs(a["conf"] - b["conf"])
    # only compare geometry where reference found something
    m = a["conf"] > 0
    dcx = np.abs(a["cx"] - b["cx"])[m]
    dcy = np.abs(a["cy"] - b["cy"])[m]
    ddiam = np.abs(a["diam"] - b["diam"])[m]
    print(f"confidence: max|d|={dconf.max():.6f} mean|d|={dconf.mean():.6e}")
    print(f"center x  : max|d|={dcx.max():.6f} mean|d|={dcx.mean():.6e}  (px, where ref conf>0)")
    print(f"center y  : max|d|={dcy.max():.6f} mean|d|={dcy.mean():.6e}")
    print(f"diameter  : max|d|={ddiam.max():.6f} mean|d|={ddiam.mean():.6e}")
    # detection agreement
    da = a["conf"] > 0; db = b["conf"] > 0
    print(f"detection agreement: {(da==db).mean()*100:.3f}%  "
          f"(ref={int(da.sum())}, cand={int(db.sum())})")
    exact = (dconf < 1e-9)
    print(f"frames bit-identical confidence: {exact.mean()*100:.2f}%")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    pr = sub.add_parser("run")
    pr.add_argument("--tag", required=True)
    pr.add_argument("--frames", default="bench/frames_2000.raw")
    pr.add_argument("--max-frames", type=int, default=None)
    pr.add_argument("--out", default=None)
    pr.set_defaults(func=run)
    pc = sub.add_parser("cmp")
    pc.add_argument("--ref", required=True)
    pc.add_argument("--cand", required=True)
    pc.set_defaults(func=cmp)
    args = p.parse_args()
    args.func(args)
