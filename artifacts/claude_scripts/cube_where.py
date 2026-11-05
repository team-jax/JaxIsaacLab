import sys; sys.argv = sys.argv[:1] + ["--headless"]
from isaaclab.app import AppLauncher
app = AppLauncher(headless=True).app
import gymnasium as gym, isaaclab_tasks  # noqa
from isaaclab_tasks.direct.one_arm_jax_minimal.v2_arm_raise import JaxV2RaiseCubeEnvCfg
c = JaxV2RaiseCubeEnvCfg(); c.scene.num_envs = 1; c.raise_cube_dx_m = 1e-6; c.raise_cube_dy_m = 1e-6; c.raise_table_dx_m = 0.0; c.raise_table_dz_m = 0.0
u = gym.make("Isaac-JaxV2-RaiseCube-Direct-v25", cfg=c).unwrapped; u.reset()
o = u.scene.env_origins[0]
print("CUBE", (u.object.data.root_pos_w[0] - o).tolist(), "ROBOT", (u.robot.data.root_pos_w[0] - o).tolist(), "TABLE", (u.table.data.root_pos_w[0] - o).tolist(), "edge_x_cfg", c.raise_table_edge_x, "table_center", c.table_center, flush=True)
import os; os._exit(0)
