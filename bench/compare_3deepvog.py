"""Compare the classical pupil_detectors Detector2D against 3DeepVOG's CNN
pupil segmentation (an independent "gold standard").

Two modes:
  gold [stride]  -- run 3DeepVOG over the clip, cache ellipse params to disk
                    (the reference is fixed, so compute it once).
  eval [tag]     -- run the current pupil_detectors over the clip (sequentially,
                    so it gets its temporal strong-prior), compare to the cached
                    gold, print agreement split by clean vs hard/blink frames,
                    and render overlays (green=pupil_detectors, red=3DeepVOG).

Run in .venv-3dvog (has torch+monai+transformers + the pupil_detectors wheel).
"""
import os, sys
os.add_dll_directory(r"c:/tools/opencv/build/x64/vc16/bin")
sys.path.insert(0, os.path.dirname(__file__))
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, ".deps", "3deepvog"))

import numpy as np
import cv2
from bench import load_frames

W, H = 640, 480
FRAMES_FILE = os.environ.get("FRAMES_FILE", "bench/frames_2000.raw")
OUT = os.environ.get("OUT_DIR", "bench/compare_frames")
GOLD = os.environ.get("GOLD_FILE", "bench/dv_gold.npz")
os.makedirs(OUT, exist_ok=True)
SHOW_IDS = [0, 250, 500, 800, 1100, 1500]  # fixed spread for the visual sheet


def fit_pupil_3dvog(prob, thr=0.5):
    """Fit ellipse to 3DeepVOG pupil prob map (channel 0). Returns cv2 RotatedRect, area."""
    mask = (prob > thr).astype(np.uint8) * 255
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    area = int((mask > 0).sum())
    if not cnts:
        return None, area
    c = max(cnts, key=cv2.contourArea)
    if len(c) < 5:
        return None, area
    return cv2.fitEllipse(c), area


def cmd_gold(stride=1):
    import torch
    from threedeepvog.models.deepvog3d_model import Model_3DeepVOG
    frames = load_frames(FRAMES_FILE)
    fids = list(range(0, len(frames), stride))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[gold] device={dev}, {len(fids)} frames")
    model = Model_3DeepVOG(device=dev, model="SegResNet_3in3out", video_width=W, video_height=H)
    cx = np.full(len(fids), np.nan); cy = np.full(len(fids), np.nan)
    MA = np.full(len(fids), np.nan); ma = np.full(len(fids), np.nan)
    ang = np.full(len(fids), np.nan); area = np.zeros(len(fids)); valid = np.zeros(len(fids), bool)
    B = 16
    import time; t0 = time.perf_counter()
    for s in range(0, len(fids), B):
        bids = fids[s:s + B]
        x = torch.from_numpy(frames[bids].astype(np.float32))
        segs = model.predict(x)
        for k, fid in enumerate(bids):
            el, a = fit_pupil_3dvog(segs[k, :, :, 0].detach().cpu().numpy())
            j = s + k; area[j] = a
            if el is not None:
                (ex, ey), (eMA, ema), eang = el
                cx[j], cy[j], MA[j], ma[j], ang[j], valid[j] = ex, ey, eMA, ema, eang, True
        if s % (B * 16) == 0:
            print(f"  {s}/{len(fids)}  ({(time.perf_counter()-t0):.0f}s)", flush=True)
    np.savez(GOLD, fids=np.array(fids), cx=cx, cy=cy, MA=MA, ma=ma, ang=ang, area=area, valid=valid)
    print(f"[gold] saved {GOLD} in {time.perf_counter()-t0:.0f}s; 3DeepVOG valid {valid.sum()}/{len(fids)}")


def cmd_eval(tag="current"):
    from pupil_detectors import Detector2D
    g = np.load(GOLD)
    fids = g["fids"]; gvalid = g["valid"]; gcx = g["cx"]; gcy = g["cy"]
    gdiam = np.maximum(g["MA"], g["ma"]); garea = g["area"]
    frames = load_frames(FRAMES_FILE)
    fidset = {int(f): i for i, f in enumerate(fids)}

    # sequential pass -> capture PD result at sampled frames (keep strong-prior)
    det = Detector2D()
    pconf = np.zeros(len(fids)); pcx = np.full(len(fids), np.nan); pcy = np.full(len(fids), np.nan)
    pdiam = np.full(len(fids), np.nan)
    for i in range(int(fids.max()) + 1):
        r = det.detect(frames[i])
        j = fidset.get(i)
        if j is not None:
            pconf[j] = r["confidence"]; pdiam[j] = r["diameter"]
            pcx[j], pcy[j] = r["ellipse"]["center"]

    conf_thr = float(os.environ.get("CONF_THR", "0"))
    pdet = pconf > conf_thr
    both = pdet & gvalid
    dist = np.hypot(pcx - gcx, pcy - gcy)
    ddiam = np.abs(pdiam - gdiam)

    # "hard" = blink-ish (3DeepVOG pupil area well below median) or either detector missing
    med_area = np.median(garea[garea > 0])
    blinkish = garea < 0.5 * med_area
    hard = blinkish | (~pdet) | (~gvalid)
    clean = both & (~hard)

    def stats(mask):
        m = mask & both
        d = dist[m]; dd = ddiam[m]
        if d.size == 0:
            return "  (no frames)"
        return (f"n={m.sum():4d}  center px: mean={d.mean():.3f} median={np.median(d):.3f} "
                f"p90={np.percentile(d,90):.3f} max={d.max():.3f} | diam |d|: mean={dd.mean():.3f}")

    print(f"\n=== [{tag}] agreement vs 3DeepVOG over {len(fids)} frames (conf>{conf_thr}) ===")
    print(f"detection: pupil_detectors {pdet.sum()}/{len(fids)} ({100*pdet.mean():.1f}%), "
          f"3DeepVOG {gvalid.sum()}/{len(fids)} ({100*gvalid.mean():.1f}%), both {both.sum()}")
    print(f"blink-ish frames (3DVOG area < 0.5*median): {blinkish.sum()}")
    print(f"  PD misses on blink-ish: {(blinkish & ~pdet).sum()}/{blinkish.sum()};  "
          f"3DVOG invalid on blink-ish: {(blinkish & ~gvalid).sum()}/{blinkish.sum()}")
    print(f"ALL  (both detect): {stats(np.ones(len(fids), bool))}")
    print(f"CLEAN              : {stats(clean)}")
    print(f"HARD/blink         : {stats(hard)}")
    print(f"disagreements >3px : {((dist > 3) & both).sum()}")

    # ---- render: fixed spread + hard cases ----
    def draw(fid):
        j = fidset[fid]
        vis = cv2.cvtColor(frames[fid], cv2.COLOR_GRAY2BGR)
        det2 = Detector2D()
        for k in range(max(0, fid - 30), fid + 1):  # short warm-up for a faithful single-frame draw
            rr = det2.detect(frames[k])
        if rr["confidence"] > 0:
            e = rr["ellipse"]
            cv2.ellipse(vis, tuple(int(v) for v in e["center"]),
                        tuple(int(v / 2) for v in e["axes"]), e["angle"], 0, 360, (0, 255, 0), 1)
        if gvalid[j]:
            cv2.ellipse(vis, ((float(gcx[j]), float(gcy[j])), (float(g["MA"][j]), float(g["ma"][j])),
                              float(g["ang"][j])), (0, 0, 255), 1)
        tag2 = "blink" if blinkish[j] else ("PD-miss" if not pdet[j] else f"d={dist[j]:.1f}px")
        cv2.putText(vis, f"f{fid} {tag2}", (8, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.putText(vis, "green=PD red=3DVOG", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        return vis

    def sheet(ids, name):
        ids = [i for i in ids if i in fidset]
        if not ids:
            return
        imgs = [draw(i) for i in ids]
        while len(imgs) % 2: imgs.append(np.zeros_like(imgs[0]))
        rows = [np.hstack(imgs[i:i + 2]) for i in range(0, len(imgs), 2)]
        cv2.imwrite(f"{OUT}/{name}", np.vstack(rows))
        print(f"  -> {OUT}/{name}  ({ids})")

    sheet(SHOW_IDS, "contact_sheet.png")
    # hardest cases: blink frames + biggest disagreements
    hard_ids = list(np.array(fids)[blinkish][:6])
    div = np.array(fids)[both]; divd = dist[both]
    hard_ids += list(div[np.argsort(-divd)[:6]])
    sheet(sorted(set(int(i) for i in hard_ids)), "hard_cases.png")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "eval"
    if mode == "gold":
        cmd_gold(int(sys.argv[2]) if len(sys.argv) > 2 else 1)
    else:
        cmd_eval(sys.argv[2] if len(sys.argv) > 2 else "current")
