"""Export an RSL-RL grasp-lift actor for the real robot: observation normalizer + MLP (ELU) as one TorchScript module
(and ONNX when the onnx package is there), plus a JSON with the normalizer, layout and action meaning.

usage: _isaac_sim/python.sh artifacts/claude_scripts/export_policy.py <checkpoint.pt> [out_dir]
The exported module maps a (N, 43) observation to the (N, 11) deterministic action (the Gaussian mean, before the
env's clip to [-1, 1] and its low-pass filter)."""
import json
import os
import sys

import torch
import torch.nn as nn

CK = sys.argv[1]
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(CK), "exported")
os.makedirs(OUT, exist_ok=True)
name = os.path.splitext(os.path.basename(CK))[0]

sd = torch.load(CK, map_location="cpu", weights_only=False)["actor_state_dict"]
EPS = 1e-2  # rsl_rl EmpiricalNormalization: (x - mean) / (std + eps)
IN = sd["mlp.0.weight"].shape[1]; HID = sd["mlp.0.weight"].shape[0]; OUTD = sd["mlp.4.weight"].shape[0]


class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("mean", sd["obs_normalizer._mean"].clone())
        self.register_buffer("std", sd["obs_normalizer._std"].clone())
        self.mlp = nn.Sequential(nn.Linear(IN, HID), nn.ELU(), nn.Linear(HID, HID), nn.ELU(), nn.Linear(HID, OUTD))
        self.mlp.load_state_dict({k[len("mlp."):]: v for k, v in sd.items() if k.startswith("mlp.")})

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.mlp((obs - self.mean) / (self.std + EPS))


actor = Actor().eval()

# check against a plain re-computation from the state dict
x = torch.randn(64, IN)
with torch.no_grad():
    h = (x - sd["obs_normalizer._mean"]) / (sd["obs_normalizer._std"] + EPS)
    h = torch.nn.functional.elu(h @ sd["mlp.0.weight"].T + sd["mlp.0.bias"])
    h = torch.nn.functional.elu(h @ sd["mlp.2.weight"].T + sd["mlp.2.bias"])
    ref = h @ sd["mlp.4.weight"].T + sd["mlp.4.bias"]
    err = float((actor(x) - ref).abs().max())
print("CHECK max |diff|", err, flush=True)

ts_path = os.path.join(OUT, f"policy_{name}.pt")
traced = torch.jit.trace(actor, torch.zeros(1, IN))
traced.save(ts_path)
with torch.no_grad():
    err_ts = float((torch.jit.load(ts_path)(x) - ref).abs().max())
print("TORCHSCRIPT", ts_path, "max |diff|", err_ts, flush=True)

onnx_path = None
try:
    import onnx  # noqa: F401
    onnx_path = os.path.join(OUT, f"policy_{name}.onnx")
    torch.onnx.export(actor, torch.zeros(1, IN), onnx_path, input_names=["obs"], output_names=["action"],
                      dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}}, opset_version=17)
    print("ONNX", onnx_path, flush=True)
except Exception as e:  # onnx not installed or export failed
    print("ONNX skipped:", repr(e), flush=True)

meta = {
    "checkpoint": CK,
    "network": f"obs ({IN}) -> (obs - mean) / (std + 0.01) -> Linear {HID} -> ELU -> Linear {HID} -> ELU -> Linear {OUTD}",
    "obs_normalizer": {"eps": EPS, "mean": sd["obs_normalizer._mean"][0].tolist(), "std": sd["obs_normalizer._std"][0].tolist()},
    "action_post": f"clip each of the {OUTD} outputs to [-1, 1], then the env's low-pass filter a = 0.25 * new + 0.75 * previous",
    "control_dt_s": 1.0 / 60.0,
    "files": {"torchscript": ts_path, "onnx": onnx_path},
}
with open(os.path.join(OUT, f"policy_{name}_meta.json"), "w") as f:
    json.dump(meta, f, indent=1)
print("META", os.path.join(OUT, f"policy_{name}_meta.json"), flush=True)
