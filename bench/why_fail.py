"""Render why-it-fails comparison crops for specific frames.

Usage (run once per mode so the process-static blur selection applies):
  PUPIL_BLUR=median   python bench/why_fail.py pd     median
  PUPIL_BLUR=gaussian python bench/why_fail.py pd     gaussian
                      python bench/why_fail.py dvog   dvog
                      python bench/why_fail.py combine
"""
import os, sys, cv2
os.add_dll_directory(r"c:/tools/opencv/build/x64/vc16/bin")
sys.path.insert(0, "bench")
sys.path.insert(0, r"c:/Users/adf44/source/python/pupil-detectors/.deps/3deepvog")
import numpy as np
from bench import load_frames

FR = load_frames("bench/frames_2000.raw")
G = np.load("bench/dv_gold.npz"); FSET = {int(f): i for i, f in enumerate(G["fids"])}
FIDS = [958, 686]   # 958: gaussian recovers, median misses; 686: both fail, CNN ok
OUT = "bench/compare_frames"
R = 95


def crop(img, fid):
    j = FSET[fid]; cx, cy = G["cx"][j], G["cy"][j]
    x0, y0 = max(0, int(cx - R)), max(0, int(cy - R))
    x1, y1 = min(640, int(cx + R)), min(480, int(cy + R))
    c = img[y0:y1, x0:x1]
    return cv2.resize(c, (c.shape[1] * 2, c.shape[0] * 2), interpolation=cv2.INTER_NEAREST)


def label(img, txt, color=(0, 255, 255)):
    cv2.putText(img, txt, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return img


def run_pd(mode):
    from pupil_detectors import Detector2D
    d = Detector2D()
    res = {}
    for i in range(max(FIDS) + 1):
        gray = FR[i]
        if i in FIDS:
            color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            r = d.detect(gray, color_img=color)  # visualize -> green=filtered edges, blue=dark mask
            res[i] = (r["confidence"], color)
        else:
            d.detect(gray)
    for fid in FIDS:
        conf, overlay = res[fid]
        c = crop(overlay, fid)
        label(c, f"{mode}  f{fid} conf={conf:.2f}")
        cv2.imwrite(f"{OUT}/why_{mode}_{fid}.png", c)
    print(f"{mode}: saved crops (green=filtered edges, blue=dark mask)")


def run_dvog():
    import torch
    from threedeepvog.models.deepvog3d_model import Model_3DeepVOG
    m = Model_3DeepVOG(device="cpu", model="SegResNet_3in3out", video_width=640, video_height=480)
    for fid in FIDS:
        x = torch.from_numpy(FR[fid][None].astype(np.float32))
        prob = m.predict(x)[0, :, :, 0].detach().cpu().numpy()
        mask = (prob > 0.5).astype(np.uint8) * 255
        base = cv2.cvtColor(FR[fid], cv2.COLOR_GRAY2BGR)
        ov = base.copy(); ov[mask > 0] = (0, 0, 255)
        vis = cv2.addWeighted(base, 0.55, ov, 0.45, 0)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if cnts:
            cc = max(cnts, key=cv2.contourArea)
            if len(cc) >= 5:
                cv2.ellipse(vis, cv2.fitEllipse(cc), (0, 255, 0), 1)
        c = crop(vis, fid)
        label(c, f"3DeepVOG f{fid}", (0, 0, 255))
        cv2.imwrite(f"{OUT}/why_dvog_{fid}.png", c)
    print("dvog: saved crops (red=pupil seg, green=ellipse)")


def combine():
    rows = []
    for fid in FIDS:
        imgs = [cv2.imread(f"{OUT}/why_{m}_{fid}.png") for m in ("median", "gaussian", "dvog")]
        h = min(i.shape[0] for i in imgs)
        imgs = [i[:h] for i in imgs]
        rows.append(np.hstack(imgs))
    w = min(r.shape[1] for r in rows)
    cv2.imwrite(f"{OUT}/why_fail.png", np.vstack([r[:, :w] for r in rows]))
    print(f"-> {OUT}/why_fail.png  (rows: f958 gaussian-recovers, f686 both-fail; cols: median | gaussian | 3DeepVOG)")


cmd = sys.argv[1]
if cmd == "pd":
    run_pd(sys.argv[2])
elif cmd == "dvog":
    run_dvog()
elif cmd == "combine":
    combine()
