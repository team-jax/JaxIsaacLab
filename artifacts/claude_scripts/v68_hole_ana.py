"""v68: compare failed and held envs of v68_hole_diag runs: argv = .pt files."""
import sys, torch
for f in sys.argv[1:]:
    d = torch.load(f); t = d["t"]; held = d["held"]; nf = int((~held).sum()); print("==", f.split("/")[-1], "held %d/%d" % (int(held.sum()), len(held)))
    def at(key, s, red=None):
        i = int((t - s).abs().argmin()); v = d[key][i]
        return v if red is None else red(v)
    for name, m in (("held", held), ("fail", ~held)):
        if not bool(m.any()):
            continue
        r = lambda v: " ".join("%.1f" % float(x) for x in v)
        yaw = [float(at("yaw", s)[m].mean()) for s in (0.5, 1.0, 1.5, 2.0, 3.0)]
        xy = [float(at("xy", s)[m].mean()) for s in (0.5, 1.0, 1.5, 2.0)]
        rise = [float(at("rise", s)[m].mean()) for s in (1.0, 2.0, 3.0, 5.0, 8.0)]
        pads1 = at("pads", 1.0)[m].mean(0); pads2 = at("pads", 2.0)[m].mean(0)
        fq1 = at("fq", 1.0)[m].mean(0); tip1 = at("tipz", 0.5)[m].mean(0)
        print("  %s n=%d | yaw@0.5/1/1.5/2/3s %s | xy mm@0.5-2s %s | rise mm@1/2/3/5/8s %s" % (name, int(m.sum()), r(yaw), r(xy), r(rise)))
        print("     pads@1s %s | pads@2s %s | fingers@1s %s | tip z - cube@0.5s mm %s | pads at start %s" % (r(pads1), r(pads2), r(fq1), r(tip1), r(d["pads_start"][m].mean(0))))
    yawend = at("yaw", 2.0)
    print("  yaw change @2s per env (fail):", " ".join("%.0f" % float(v) for v in yawend[~held][:20]), "| held:", " ".join("%.0f" % float(v) for v in yawend[held][:20]))
