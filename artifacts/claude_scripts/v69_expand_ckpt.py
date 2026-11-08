"""v69: copy a 43-input grasp-lift checkpoint to a 45-input one (GL_OBS_YAW): actor and critic first layers get two zero
input columns (the policy acts exactly as before until training uses them), the observation normalizers mean 0 / var 0.5
for sin / cos, the Adam moments of those weights zero-padded. argv: src.pt dst.pt"""
import sys, os, torch
src, dst = sys.argv[1], sys.argv[2]
d = torch.load(src, map_location="cpu", weights_only=False)
ADD = 2
for part in ("actor_state_dict", "critic_state_dict"):
    sd = d[part]
    w = sd["mlp.0.weight"]; sd["mlp.0.weight"] = torch.cat((w, torch.zeros(w.shape[0], ADD)), dim=1)
    sd["obs_normalizer._mean"] = torch.cat((sd["obs_normalizer._mean"], torch.zeros(1, ADD)), dim=1)
    sd["obs_normalizer._var"] = torch.cat((sd["obs_normalizer._var"], torch.full((1, ADD), 0.5)), dim=1)
    sd["obs_normalizer._std"] = torch.cat((sd["obs_normalizer._std"], torch.full((1, ADD), 0.5 ** 0.5)), dim=1)
n_pad = 0
st = d["optimizer_state_dict"]["state"]
for k, v in st.items():
    for m in ("exp_avg", "exp_avg_sq"):
        if m in v and v[m].dim() == 2 and v[m].shape[1] == 43:
            v[m] = torch.cat((v[m], torch.zeros(v[m].shape[0], ADD, dtype=v[m].dtype)), dim=1); n_pad += 1
os.makedirs(os.path.dirname(dst), exist_ok=True)
torch.save(d, dst)
print("EXPANDED", dst, "padded moments", n_pad, "actor in", tuple(d["actor_state_dict"]["mlp.0.weight"].shape), "critic in", tuple(d["critic_state_dict"]["mlp.0.weight"].shape))
