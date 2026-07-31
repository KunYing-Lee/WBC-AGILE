# SPDX-FileCopyrightText: Copyright (c) 2026 KunYing Lee
# SPDX-License-Identifier: Apache-2.0

"""Contract tests for the Booster K1 omnidirectional locomotion task."""

from agile.rl_env.assets.robots import booster_k1
from agile.rl_env.tasks.locomotion.k1.velocity_env_cfg import (
    K1_ANG_VEL_Z_RANGE,
    K1_LIN_VEL_X_RANGE,
    K1_LIN_VEL_Y_RANGE,
    ActionsCfg,
    CommandsCfg,
    K1LowerVelocityEnvCfg,
)


def test_k1_velocity_command_bounds() -> None:
    commands = CommandsCfg().base_velocity
    assert tuple(commands.ranges.lin_vel_x) == K1_LIN_VEL_X_RANGE
    assert tuple(commands.ranges.lin_vel_y) == K1_LIN_VEL_Y_RANGE
    assert tuple(commands.ranges.ang_vel_z) == K1_ANG_VEL_Z_RANGE


def test_k1_locomotion_controls_exact_leg_order() -> None:
    action = ActionsCfg().joint_pos
    assert action.joint_names == booster_k1.K1_LEG_JOINT_NAMES
    assert action.preserve_order is True
    assert action.scale == 0.25
    assert len(action.joint_names) == 12


def test_k1_locomotion_uses_deployed_neutral_leg_posture() -> None:
    joint_pos = booster_k1.K1_LOCOMOTION_CFG.init_state.joint_pos
    assert joint_pos["Left_Hip_Pitch"] == -0.15
    assert joint_pos["Left_Knee_Pitch"] == 0.3
    assert joint_pos["Left_Ankle_Pitch"] == -0.15
    assert joint_pos["Right_Hip_Pitch"] == -0.15
    assert joint_pos["Right_Knee_Pitch"] == 0.3
    assert joint_pos["Right_Ankle_Pitch"] == -0.15


def test_k1_locomotion_rates_match_wbc_agile_contract() -> None:
    env = K1LowerVelocityEnvCfg()
    assert env.controller_freq == 50.0
    assert env.physics_freq == 200.0
    assert env.decimation == 4
