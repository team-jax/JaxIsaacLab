"""Transplant a v8 (40 observations) RSL-RL checkpoint into the v9 (43) layout.

v8 observation: ... [34:37] cube - hand, [37:40] second-link pressure.
v9 observation: ... [34:37] cube - hand, [37:40] first-link pressure,
[40:43] second-link pressure (RA12P model). The first layers of actor and
critic (and their Adam moments) are widened: v8 columns 0-36 stay in place,
v8 columns 37-39 move to 40-42, and the new first-link columns start at zero
weight, so before any contact the policy acts exactly as the v8 policy.
Observation normalizer stats follow the same mapping (new columns: mean 0,
var 1). Usage: python convert_v8_to_v9_checkpoint.py <v8 model.pt> <out .pt>
"""
import sys

import torch

OLD, NEW = 40, 43
OLD_COLS = list(range(37)) + [37, 38, 39]
NEW_COLS = list(range(37)) + [40, 41, 42]


def widen(t: torch.Tensor, fill: float) -> torch.Tensor:
    out = torch.full((*t.shape[:-1], NEW), fill, dtype=t.dtype)
    out[..., NEW_COLS] = t[..., OLD_COLS]
    return out


def main(src: str, dst: str) -> None:
    ck = torch.load(src, map_location="cpu", weights_only=False)
    for part in ("actor_state_dict", "critic_state_dict"):
        sd = ck[part]
        sd["mlp.0.weight"] = widen(sd["mlp.0.weight"], 0.0)
        sd["obs_normalizer._mean"] = widen(sd["obs_normalizer._mean"], 0.0)
        sd["obs_normalizer._var"] = widen(sd["obs_normalizer._var"], 1.0)
        sd["obs_normalizer._std"] = widen(sd["obs_normalizer._std"], 1.0)
    for state in ck["optimizer_state_dict"]["state"].values():
        for key in ("exp_avg", "exp_avg_sq"):
            if key in state and state[key].dim() == 2 and state[key].shape[-1] == OLD:
                state[key] = widen(state[key], 0.0)
    torch.save(ck, dst)
    print(f"converted {src} -> {dst}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
