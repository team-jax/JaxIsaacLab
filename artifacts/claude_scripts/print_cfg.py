import sys; sys.argv=['x']
from isaaclab.app import AppLauncher; app=AppLauncher(headless=True).app
from isaaclab_tasks.direct.one_arm_jax_minimal.v2_front_pregrasp import JaxV2FrontPregraspEnvCfg as C
from isaaclab_tasks.direct.one_arm_jax_minimal.v2_front_grasp_close import JaxV2FrontGraspCloseEnvCfg as G
c=C(); g=G()
print('EFFECTIVE', {k: getattr(c, k) for k in ('pre_height_offset_m','pre_plate_scale','pre_plate_mesh','pre_rotation_start_s','pre_rotation_ramp_s','pre_rotation_delta_deg','pre_hide_scripted_obs','pre_target_dx_m','pre_table_force_scale','pre_ref_q_deg','pre_ref_scale','pre_ref_after_s')}, flush=True)
print('EFFECTIVE_V13', {k: getattr(g, k) for k in ('pre_policy_path','grasp_guide_deg','close_rotation_delta_deg','close_force_balance')}, flush=True)
import os; os._exit(0)
