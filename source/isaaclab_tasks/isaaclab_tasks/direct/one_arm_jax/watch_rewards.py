"""Print the reward terms of an existing skrl run in a separate terminal."""

import argparse
import glob
import os
import time

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


parser = argparse.ArgumentParser()
parser.add_argument("--run-dir", default="", help="Run directory; defaults to the newest lift-hold PPO run")
parser.add_argument("--interval", type=float, default=10.0, help="Seconds between checks")
parser.add_argument("--once", action="store_true", help="Print the latest values and exit")
args = parser.parse_args()

if args.run_dir:
    run_dir = args.run_dir
else:
    runs = glob.glob("logs/skrl/one_arm_jax_lift_hold_reward_v1/*_ppo_torch")
    runs += glob.glob("logs/skrl/one_arm_jax_lift_hold/*_ppo_torch")
    if not runs:
        parser.error("No lift-hold PPO run was found")
    run_dir = max(runs, key=os.path.getmtime)

reader = EventAccumulator(run_dir, size_guidance={"scalars": 0})
last_step = -1
print(f"Reading {run_dir}", flush=True)

while True:
    reader.Reload()
    available = set(reader.Tags().get("scalars", []))
    if "Info / reach" in available:
        step = reader.Scalars("Info / reach")[-1].step
        if step != last_step:
            last_step = step

            def value(tag):
                return reader.Scalars(tag)[-1].value if tag in available else float("nan")

            print(
                f"it={step // 24:5d}  R={value('Reward / Instantaneous reward (mean)'):.3f}"
                f"  reach={value('Info / reach'):.3f}"
                f"  contactQ={value('Info / contact_quality_reward'):.3f}"
                f"  geometry={value('Info / contact_geometry_reward'):.3f}"
                f"  graspQ={value('Info / grasp_quality'):.3f}",
                flush=True,
            )
            print(
                f"          graspR={value('Info / grasp_reward'):.3f}"
                f"  liftR={value('Info / lift_reward'):.3f}"
                f"  holdR={value('Info / hold_reward'):.3f}"
                f"  success={value('Info / success'):.3f}"
                f"  excess={value('Info / pressure_excess_penalty'):.3f}"
                f"  drop={value('Info / drop_penalty'):.3f}"
                f"  action={value('Info / action_rate_penalty'):.3f}",
                flush=True,
            )
    if args.once:
        break
    time.sleep(args.interval)
