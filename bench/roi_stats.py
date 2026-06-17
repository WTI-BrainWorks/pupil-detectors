import os, sys, subprocess
import numpy as np

PY = r"c:/Users/adf44/source/python/pupil-detectors/.venv-3dvog/Scripts/python.exe"
SNIP = '''
import os,sys; os.add_dll_directory(r"c:/tools/opencv/build/x64/vc16/bin"); sys.path.insert(0,"bench")
from bench import load_frames
from pupil_detectors import Detector2D
fr=load_frames(r"bench/%s.raw"); d=Detector2D()
for i in range(len(fr)): d.detect(fr[i])
'''

for seg in ("frames_2000", "frames_blink"):
    e = dict(os.environ); e["PUPIL_ROI_LOG"] = "1"
    out = subprocess.run([PY, "-c", SNIP % seg], capture_output=True, text=True, env=e)
    ws, hs = [], []
    for ln in out.stderr.splitlines():
        if ln.startswith("ROI"):
            _, w, h = ln.split(); ws.append(int(w)); hs.append(int(h))
    ws = np.array(ws); hs = np.array(hs); area = ws * hs
    if len(ws) == 0:
        print(seg, "no ROI logs; stderr tail:", out.stderr[-300:]); continue
    print(f"{seg}: n={len(ws)}  coarse-ROI fired {len(ws)}x")
    print(f"  width : med={np.median(ws):.0f} p10={np.percentile(ws,10):.0f} p90={np.percentile(ws,90):.0f} max={ws.max()}")
    print(f"  height: med={np.median(hs):.0f} p10={np.percentile(hs,10):.0f} p90={np.percentile(hs,90):.0f} max={hs.max()}")
    print(f"  area  : med={np.median(area):.0f}  (full 640x480=307200 -> ROI is {100*np.median(area)/307200:.1f}% of frame)")
