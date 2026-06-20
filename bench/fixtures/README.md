# Benchmark fixtures

`eye_frames_ci.npz` — 12 grayscale eye-camera frames (640×480, PNG-encoded
inside the npz), sampled from a clean fixation segment of the shared sample
video. Used by [`../ci_bench.py`](../ci_bench.py) for the CI benchmark of the
built wheel against the PyPI release. Kept small (~1.4 MB) so it can live in the
repo and run network-free.

## Sample data source

The frames were extracted from `eye1.mp4` in the Pupil Labs eye-camera
recordings. The full public dataset (Eyelink 1000 data + Pupil Labs videos,
~700 GB) is on figshare:

> Eyelink 1000 data and Pupil Labs Videos (700GB)
> https://figshare.com/collections/Eyelink_1000_data_and_Pupil_Labs_Videos_700GB_/4379810/2

Regenerate the fixture from a local copy of the video with the snippet documented
at the top of [`../ci_bench.py`](../ci_bench.py) (decode N frames, `cv2.imencode`
to PNG, `np.savez_compressed`).
