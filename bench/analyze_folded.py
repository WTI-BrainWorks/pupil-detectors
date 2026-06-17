import re, sys, collections

path = sys.argv[1] if len(sys.argv) > 1 else "bench/profile.folded"
mod_self = collections.Counter()   # self time by module
sym_self = collections.Counter()   # self time by leaf symbol
total = 0

def modof(frame):
    m = re.search(r"\(([^)]+)\)\s*$", frame)
    mod = m.group(1) if m else "?"
    # collapse module path to basename
    mod = mod.replace("\\", "/").split("/")[-1]
    return mod

with open(path) as f:
    for line in f:
        line = line.rstrip("\n")
        if not line:
            continue
        try:
            stack, cnt = line.rsplit(" ", 1)
            cnt = int(cnt)
        except ValueError:
            continue
        total += cnt
        frames = stack.split(";")
        leaf = frames[-1]
        mod_self[modof(leaf)] += cnt
        sym_self[leaf] += cnt

print(f"total samples: {total}\n")
print("=== SELF time by module ===")
for mod, c in mod_self.most_common(15):
    print(f"{c:8d}  {100*c/total:5.1f}%  {mod}")

print("\n=== SELF time by leaf symbol (named only, top 30) ===")
shown = 0
for sym, c in sym_self.most_common(200):
    if sym.strip().startswith("0x"):
        continue
    print(f"{c:8d}  {100*c/total:5.1f}%  {sym}")
    shown += 1
    if shown >= 30:
        break

# Aggregate opencv self-time grouped by nearest named ancestor in detector (attribution)
print("\n=== OpenCV self-time attributed to calling detector frame ===")
attrib = collections.Counter()
with open(path) as f:
    for line in f:
        line = line.rstrip("\n")
        if not line:
            continue
        try:
            stack, cnt = line.rsplit(" ", 1); cnt = int(cnt)
        except ValueError:
            continue
        frames = stack.split(";")
        leaf = frames[-1]
        if "opencv_world" in leaf or "cv2.pyd" in leaf:
            # find nearest ancestor that is a named (non-0x) detector_2d or python frame
            tag = None
            for fr in reversed(frames[:-1]):
                if "detector_2d" in fr and not fr.strip().startswith("0x"):
                    tag = fr; break
                if ".pyx" in fr or ".py:" in fr:
                    tag = fr; break
            attrib[tag or "<unknown call site>"] += cnt
for k, c in attrib.most_common(20):
    print(f"{c:8d}  {100*c/total:5.1f}%  <- {k}")
