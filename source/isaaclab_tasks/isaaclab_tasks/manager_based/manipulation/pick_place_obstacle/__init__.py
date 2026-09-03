# SPDX-License-Identifier: BSD-3-Clause
"""Pick-and-place + obstacle avoidance RL task.

Training command
----------------
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \\
        --task Isaac-PickPlace-Obstacle-Franka-v0 \\
        --num_envs 1024 \\
        --headless

Play/eval command
-----------------
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \\
        --task Isaac-PickPlace-Obstacle-Franka-Play-v0 \\
        --num_envs 16 \\
        --checkpoint logs/rsl_rl/franka_pick_place_obstacle/...

Debug eval with EE trajectory logging
--------------------------------------
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_debug_ee.py \\
        --task Isaac-PickPlace-Obstacle-Franka-Play-v0 \\
        --num_envs 1 \\
        --checkpoint logs/rsl_rl/franka_pick_place_obstacle/... \\
        --verbose_ee --log_csv logs/ee_traj.csv
"""

from .config.franka import *  # noqa: F401, F403
