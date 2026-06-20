"""Compare two ci_bench.py JSON outputs (built wheel vs PyPI release).

    python bench/ci_compare.py --built built.json --baseline pypi.json

Prints a GitHub-flavoured-markdown summary (append it to $GITHUB_STEP_SUMMARY in
CI). Exit code is 0 unless --fail-under is given and the speedup falls below it,
which lets the job optionally guard against a real perf regression while
tolerating runner noise by default.
"""
import argparse
import json


def fmt(x):
    return f"{x:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--built", required=True)
    ap.add_argument("--baseline", required=True)
    ap.add_argument(
        "--fail-under",
        type=float,
        default=None,
        help="fail if median speedup (baseline/built) drops below this ratio",
    )
    args = ap.parse_args()

    built = json.load(open(args.built))
    base = json.load(open(args.baseline))

    bl = built["latency_ms"]
    pl = base["latency_ms"]
    speedup = pl["median"] / bl["median"] if bl["median"] else float("nan")

    lines = []
    lines.append("### 2D detector benchmark — built wheel vs PyPI")
    lines.append("")
    lines.append(
        f"`{built['version']}` (built) vs `{base['version']}` (PyPI) — "
        f"{built['n_samples']} samples "
        f"({built['n_frames']} frames × {built['repeats']} repeats), same runner."
    )
    lines.append("")
    lines.append("| latency (ms) | built | PyPI | speedup |")
    lines.append("|---|---|---|---|")
    for k in ("median", "p90", "p99", "mean"):
        s = pl[k] / bl[k] if bl[k] else float("nan")
        lines.append(f"| {k} | {fmt(bl[k])} | {fmt(pl[k])} | {fmt(s)}× |")
    lines.append("")
    bs, ps = built["signature"], base["signature"]
    lines.append("| output signature | built | PyPI |")
    lines.append("|---|---|---|")
    lines.append(
        f"| mean confidence | {fmt(bs['mean_confidence'])} | {fmt(ps['mean_confidence'])} |"
    )
    lines.append(
        f"| mean diameter (px) | {fmt(bs['mean_diameter'])} | {fmt(ps['mean_diameter'])} |"
    )
    lines.append("")
    lines.append(f"**Median speedup: {fmt(speedup)}×**")

    report = "\n".join(lines)
    print(report)

    if args.fail_under is not None and speedup < args.fail_under:
        raise SystemExit(
            f"::error::median speedup {speedup:.2f}x < required {args.fail_under}x"
        )


if __name__ == "__main__":
    main()
