"""Tiny CI benchmark: time Detector2D over a committed frame fixture.

Runs against whatever `pupil_detectors` is importable in the current env, so CI
can run it once in an env with the freshly-built wheel and once in an env with
the PyPI release, then diff the two JSON outputs with ci_compare.py.

    python bench/ci_bench.py --out built.json --repeats 20

The fixture (bench/fixtures/eye_frames_ci.npz) is a dozen real grayscale eye
frames (PNG-encoded) sampled from the shared sample video. Timing on hosted CI
runners is noisy in absolute terms, so ci_compare reports the built/PyPI *ratio*
measured back-to-back on the same runner, not absolute numbers.
"""
import argparse
import json
import os
import statistics
import time

# Local convenience: a raw (un-repaired) wheel links the system OpenCV DLLs.
# Harmless in CI, where the repaired wheel bundles its own OpenCV.
_OPENCV_BIN = os.environ.get("PUPIL_OPENCV_BIN")
if _OPENCV_BIN and os.path.isdir(_OPENCV_BIN):
    os.add_dll_directory(_OPENCV_BIN)

import cv2
import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
FIXTURE = os.path.join(_THIS, "fixtures", "eye_frames_ci.npz")


def load_frames():
    data = np.load(FIXTURE, allow_pickle=True)
    return [
        cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_GRAYSCALE)
        for b in data["pngs"]
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--repeats", type=int, default=20)
    args = ap.parse_args()

    import pupil_detectors
    from pupil_detectors import Detector2D

    frames = load_frames()
    det = Detector2D()

    # warm up (first call pays one-time allocation/JIT-ish costs)
    for fr in frames:
        det.detect(fr)

    times_ms = []
    confidences = []
    diameters = []
    for _ in range(args.repeats):
        for fr in frames:
            t0 = time.perf_counter()
            res = det.detect(fr)
            times_ms.append((time.perf_counter() - t0) * 1e3)
            confidences.append(float(res["confidence"]))
            diameters.append(float(res["diameter"]))

    times_ms.sort()

    def pct(p):
        return times_ms[min(len(times_ms) - 1, int(p / 100 * len(times_ms)))]

    result = {
        "package": "pupil_detectors",
        "version": pupil_detectors.__version__,
        "n_frames": len(frames),
        "repeats": args.repeats,
        "n_samples": len(times_ms),
        "latency_ms": {
            "median": statistics.median(times_ms),
            "p90": pct(90),
            "p99": pct(99),
            "mean": statistics.fmean(times_ms),
        },
        # output signature: a detection-quality regression shows up here even
        # when timing looks fine.
        "signature": {
            "mean_confidence": statistics.fmean(confidences),
            "mean_diameter": statistics.fmean(diameters),
        },
    }
    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
