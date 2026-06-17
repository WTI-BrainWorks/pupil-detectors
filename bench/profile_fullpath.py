import os, sys, time
os.add_dll_directory(r"c:/tools/opencv/build/x64/vc16/bin")
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from bench import load_frames

frames = load_frames("bench/frames_2000.raw")
# Shuffle so consecutive frames are dissimilar -> strong-prior usually fails ->
# exercises the full contour/combinatorial detection path (the latency tail).
rng = np.random.default_rng(0)
order = rng.permutation(len(frames))
from pupil_detectors import Detector2D
det = Detector2D()
passes = int(sys.argv[1]) if len(sys.argv) > 1 else 30
print(f"full-path profiling {passes} passes x {len(frames)} frames (shuffled)", flush=True)
t0 = time.perf_counter()
for p in range(passes):
    for i in order:
        det.detect(frames[i])
print(f"done {time.perf_counter()-t0:.1f}s", flush=True)
