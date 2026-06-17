"""Compare the classical pupil_detectors Detector2D against 3DeepVOG's CNN
pupil segmentation on a handful of frames, and render side-by-side overlays.

  green  = pupil_detectors (current algorithm)
  red    = 3DeepVOG (SegResNet pupil segmentation -> ellipse fit)

Run in the 3DeepVOG venv (.venv-3dvog), which also has pupil_detectors installed.
"""
import os, sys
os.add_dll_directory(r"c:/tools/opencv/build/x64/vc16/bin")
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, r"c:/Users/adf44/source/python/pupil-detectors/.deps/3deepvog")

import numpy as np
import cv2
import torch
from bench import load_frames
from pupil_detectors import Detector2D
from threedeepvog.models.deepvog3d_model import Model_3DeepVOG

W, H = 640, 480
OUT = "bench/compare_frames"
os.makedirs(OUT, exist_ok=True)

# frames spread across the clip (varied content)
FRAME_IDS = [0, 250, 500, 800, 1100, 1500]

frames = load_frames("bench/frames_2000.raw")


def fit_pupil_3dvog(prob, thr=0.5):
    """Fit an ellipse to 3DeepVOG's pupil probability map (channel 0)."""
    mask = (prob > thr).astype(np.uint8) * 255
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None, 0
    c = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if len(c) < 5 or area < 10:
        return None, area
    return cv2.fitEllipse(c), area


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={dev}")
    model = Model_3DeepVOG(device=dev, model="SegResNet_3in3out",
                           video_width=W, video_height=H)
    det = Detector2D()

    # Run pupil_detectors sequentially so it gets its temporal strong-prior
    # (its real streaming mode); capture the result at each comparison frame.
    pd_results = {}
    want = set(FRAME_IDS)
    for i in range(max(FRAME_IDS) + 1):
        r = det.detect(frames[i])
        if i in want:
            pd_results[i] = r

    rows = []
    for fid in FRAME_IDS:
        gray = frames[fid]
        # --- pupil_detectors (from sequential pass) ---
        r = pd_results[fid]
        e = r["ellipse"]
        pd_center = e["center"]; pd_axes = e["axes"]; pd_angle = e["angle"]
        pd_conf = r["confidence"]; pd_diam = r["diameter"]

        # --- 3DeepVOG ---
        x = torch.from_numpy(gray[None].astype(np.float32))  # (1,H,W)
        segs = model.predict(x)  # (1,H,W,3)
        pupil_prob = segs[0, :, :, 0].detach().cpu().numpy()
        dv_ellipse, dv_area = fit_pupil_3dvog(pupil_prob)

        # --- draw ---
        vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        if pd_conf > 0:
            cv2.ellipse(vis, tuple(int(v) for v in pd_center),
                        tuple(int(v / 2) for v in pd_axes), pd_angle, 0, 360, (0, 255, 0), 1)
            cv2.circle(vis, tuple(int(v) for v in pd_center), 2, (0, 255, 0), -1)
        if dv_ellipse is not None:
            cv2.ellipse(vis, dv_ellipse, (0, 0, 255), 1)
            cv2.circle(vis, tuple(int(v) for v in dv_ellipse[0]), 2, (0, 0, 255), -1)

        dv_center = dv_ellipse[0] if dv_ellipse else (np.nan, np.nan)
        dv_diam = max(dv_ellipse[1]) if dv_ellipse else np.nan
        dcx = (pd_center[0] - dv_center[0]) if dv_ellipse and pd_conf > 0 else np.nan
        dcy = (pd_center[1] - dv_center[1]) if dv_ellipse and pd_conf > 0 else np.nan
        ddist = float(np.hypot(dcx, dcy)) if dv_ellipse and pd_conf > 0 else np.nan

        cv2.putText(vis, "green=pupil_detectors  red=3DeepVOG", (8, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.putText(vis, f"frame {fid}", (8, H - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
        path = f"{OUT}/frame_{fid:04d}.png"
        cv2.imwrite(path, vis)

        rows.append((fid, pd_center, pd_diam, pd_conf, dv_center, dv_diam, ddist))
        print(f"frame {fid:4d}: PD center=({pd_center[0]:.1f},{pd_center[1]:.1f}) "
              f"diam={pd_diam:.1f} conf={pd_conf:.2f} | "
              f"3DVOG center=({dv_center[0]:.1f},{dv_center[1]:.1f}) diam={dv_diam:.1f} "
              f"| center dist={ddist:.1f}px -> {path}")

    # composite contact sheet (2 cols x 3 rows)
    imgs = [cv2.imread(f"{OUT}/frame_{fid:04d}.png") for fid in FRAME_IDS]
    rowsimg = [np.hstack(imgs[i:i + 2]) for i in range(0, len(imgs), 2)]
    sheet = np.vstack(rowsimg)
    cv2.imwrite(f"{OUT}/contact_sheet.png", sheet)
    print(f"\ncontact sheet -> {OUT}/contact_sheet.png")

    # --- aggregate agreement vs gold standard (the yardstick for algorithm changes) ---
    n_total = len(frames)
    sample = list(range(0, min(1800, n_total), 12))
    sset = set(sample)
    pd_cap = {}
    det2 = Detector2D()
    for i in range(max(sample) + 1):
        rr = det2.detect(frames[i])
        if i in sset:
            pd_cap[i] = rr
    # 3DeepVOG batched
    dv = {}
    B = 16
    for s in range(0, len(sample), B):
        batch_ids = sample[s:s + B]
        x = torch.from_numpy(frames[batch_ids].astype(np.float32))
        segs = model.predict(x)
        for k, fid in enumerate(batch_ids):
            el, _ = fit_pupil_3dvog(segs[k, :, :, 0].detach().cpu().numpy())
            dv[fid] = el

    dists, ddiam = [], []
    pd_det = dvog_det = both = 0
    for fid in sample:
        r = pd_cap[fid]; el = dv[fid]
        pdok = r["confidence"] > 0; dvok = el is not None
        pd_det += pdok; dvog_det += dvok
        if pdok and dvok:
            both += 1
            pc = r["ellipse"]["center"]
            dists.append(float(np.hypot(pc[0] - el[0][0], pc[1] - el[0][1])))
            ddiam.append(abs(r["diameter"] - max(el[1])))
    dists = np.array(dists); ddiam = np.array(ddiam)
    print(f"\n=== AGREEMENT vs 3DeepVOG over {len(sample)} frames (baseline yardstick) ===")
    print(f"detection rate: pupil_detectors {pd_det}/{len(sample)} ({100*pd_det/len(sample):.1f}%), "
          f"3DeepVOG {dvog_det}/{len(sample)} ({100*dvog_det/len(sample):.1f}%), both {both}")
    print(f"center distance (px): mean={dists.mean():.3f} median={np.median(dists):.3f} "
          f"p90={np.percentile(dists,90):.3f} max={dists.max():.3f}")
    print(f"diameter |diff| (px): mean={ddiam.mean():.3f} median={np.median(ddiam):.3f} "
          f"p90={np.percentile(ddiam,90):.3f}")


if __name__ == "__main__":
    main()
