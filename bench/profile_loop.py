import os, sys, time
_OPENCV_BIN = r"c:/tools/opencv/build/x64/vc16/bin"
if os.path.isdir(_OPENCV_BIN):
    os.add_dll_directory(_OPENCV_BIN)
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from bench import load_frames

frames = load_frames("bench/frames_2000.raw")
from pupil_detectors import Detector2D
det = Detector2D()
passes = int(sys.argv[1]) if len(sys.argv) > 1 else 8
print(f"profiling {passes} passes x {len(frames)} frames", flush=True)
t0 = time.perf_counter()
for p in range(passes):
    for i in range(len(frames)):
        det.detect(frames[i])
print(f"done {time.perf_counter()-t0:.1f}s", flush=True)
