"""v71: copy a checkpoint with the action noise capped: std of every action = min(old, STD) (a0: min(old, STD0)); stds already
small stay. The policy mean (deterministic behaviour) is unchanged. argv: src.pt dst.pt STD STD0"""
import sys, os, math, torch
src, dst, std, std0 = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
d = torch.load(src, map_location="cpu", weights_only=False)
ls = d["actor_state_dict"]["distribution.log_std_param"]
old = [round(math.exp(float(v)), 3) for v in ls]
cap = torch.full_like(ls, math.log(std)); cap[0] = math.log(std0)
new = torch.minimum(ls, cap)
d["actor_state_dict"]["distribution.log_std_param"] = new
os.makedirs(os.path.dirname(dst), exist_ok=True); torch.save(d, dst)
print("STD_SET", dst, "old", old, "new", [round(math.exp(float(v)), 3) for v in new])
