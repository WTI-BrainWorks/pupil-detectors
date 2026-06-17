"""Three-way ellipse overlay: median (yellow) vs gaussian (green) vs 3DeepVOG (red).
Run: PUPIL_BLUR=median python bench/three_way.py pd median
     PUPIL_BLUR=gaussian python bench/three_way.py pd gaussian
                         python bench/three_way.py dvog
                         python bench/three_way.py combine
"""
import os, sys, cv2
os.add_dll_directory(r"c:/tools/opencv/build/x64/vc16/bin")
sys.path.insert(0, "bench")
sys.path.insert(0, r"c:/Users/adf44/source/python/pupil-detectors/.deps/3deepvog")
import numpy as np
from bench import load_frames

FR = load_frames("bench/frames_2000.raw")
FIDS = [250, 958]   # normal (all agree) ; blur-recovered (median misses)
OUT = "bench/compare_frames"; R = 90


def crop(img, cx, cy):
    x0, y0 = max(0, int(cx - R)), max(0, int(cy - R))
    x1, y1 = min(640, int(cx + R)), min(480, int(cy + R))
    c = img[y0:y1, x0:x1]
    return cv2.resize(c, (c.shape[1] * 3, c.shape[0] * 3), interpolation=cv2.INTER_LINEAR)


def run_pd(name):
    from pupil_detectors import Detector2D
    d = Detector2D(); rows = []
    for i in range(max(FIDS) + 1):
        r = d.detect(FR[i])
        if i in FIDS:
            e = r["ellipse"]
            rows.append((i, r["confidence"], e["center"][0], e["center"][1],
                         e["axes"][0], e["axes"][1], e["angle"]))
    np.save(f"bench/tw_{name}.npy", np.array(rows, float))


def run_dvog():
    import torch
    from threedeepvog.models.deepvog3d_model import Model_3DeepVOG
    m = Model_3DeepVOG(device="cpu", model="SegResNet_3in3out", video_width=640, video_height=480)
    rows = []
    for fid in FIDS:
        prob = m.predict(torch.from_numpy(FR[fid][None].astype(np.float32)))[0, :, :, 0].numpy()
        mask = (prob > 0.5).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if cnts and len(max(cnts, key=cv2.contourArea)) >= 5:
            (cx, cy), (a, b), ang = cv2.fitEllipse(max(cnts, key=cv2.contourArea))
            rows.append((fid, 1.0, cx, cy, a, b, ang))
        else:
            rows.append((fid, 0, 0, 0, 0, 0, 0))
    np.save("bench/tw_dvog.npy", np.array(rows, float))


def combine():
    med = {int(r[0]): r for r in np.load("bench/tw_median.npy")}
    gau = {int(r[0]): r for r in np.load("bench/tw_gaussian.npy")}
    dv = {int(r[0]): r for r in np.load("bench/tw_dvog.npy")}
    out = []
    for fid in FIDS:
        vis = cv2.cvtColor(FR[fid], cv2.COLOR_GRAY2BGR)
        cxc, cyc = dv[fid][2], dv[fid][3]
        for src, col, semi in [(med, (0, 230, 230), True), (gau, (0, 220, 0), True), (dv, (0, 0, 255), False)]:
            _, conf, cx, cy, a, b, ang = src[fid]
            if conf <= 0:
                continue
            # pupil_detectors axes are full lengths; 3DeepVOG (a,b) from fitEllipse are full too
            ax = (int(a / 2), int(b / 2)) if semi else (int(a / 2), int(b / 2))
            cv2.ellipse(vis, (int(cx), int(cy)), ax, ang, 0, 360, col, 1)
        c = crop(vis, cxc, cyc)
        med_txt = "miss" if med[fid][1] <= 0 else f"{med[fid][1]:.2f}"
        cv2.putText(c, f"f{fid}  median(Y)={med_txt} gauss(G)={gau[fid][1]:.2f} 3DVOG(R)", (6, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        out.append(c)
    h = min(i.shape[0] for i in out); w = min(i.shape[1] for i in out)
    cv2.imwrite(f"{OUT}/three_way.png", np.vstack([i[:h, :w] for i in out]))
    print(f"-> {OUT}/three_way.png")


cmd = sys.argv[1]
if cmd == "pd":
    run_pd(sys.argv[2])
elif cmd == "dvog":
    run_dvog()
else:
    combine()
