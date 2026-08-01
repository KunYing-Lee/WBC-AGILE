# SPDX-FileCopyrightText: Copyright (c) 2026 KunYing Lee
# SPDX-License-Identifier: Apache-2.0

"""Booster K1 22-DoF articulation and hardware-derived actuator contract."""

from __future__ import annotations

import os
from dataclasses import dataclass
from math import pi
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets.articulation import ArticulationCfg

from agile.rl_env.mdp.actuators import BoosterDelayedPDActuatorCfg

K1_HEAD_JOINT_NAMES = ["AAHead_yaw", "Head_pitch"]
K1_ARM_JOINT_NAMES = [
    "ALeft_Shoulder_Pitch",
    "Left_Shoulder_Roll",
    "Left_Elbow_Pitch",
    "Left_Elbow_Yaw",
    "ARight_Shoulder_Pitch",
    "Right_Shoulder_Roll",
    "Right_Elbow_Pitch",
    "Right_Elbow_Yaw",
]
K1_LEG_JOINT_NAMES = [
    "Left_Hip_Pitch",
    "Left_Hip_Roll",
    "Left_Hip_Yaw",
    "Left_Knee_Pitch",
    "Left_Ankle_Pitch",
    "Left_Ankle_Roll",
    "Right_Hip_Pitch",
    "Right_Hip_Roll",
    "Right_Hip_Yaw",
    "Right_Knee_Pitch",
    "Right_Ankle_Pitch",
    "Right_Ankle_Roll",
]
K1_JOINT_NAMES = K1_HEAD_JOINT_NAMES + K1_ARM_JOINT_NAMES + K1_LEG_JOINT_NAMES

HEAD_JOINT_NAMES = [".*Head.*"]
ARM_JOINT_NAMES = [".*Shoulder.*", ".*Elbow.*"]
LEG_JOINT_NAMES = [".*Hip.*", ".*Knee.*", ".*Ankle.*"]
FEET_LINK_NAMES = [".*foot_link.*"]
UNDESIRED_CONTACTS_LINKS = [
    "Trunk",
    "Head_.*",
    ".*_Arm_.*",
    ".*_hand_link",
    ".*_Hip_.*",
    ".*_Shank",
]
DEFAULT_TRUNK_HEIGHT = 0.57
LIFT_LINK_NAME = "Head_2"
MIN_DELAY_STEPS = 2
MAX_DELAY_STEPS = 8
_DEFAULT_K1_URDF_PATH = Path(__file__).resolve().parents[4] / "assets/booster_assets/robots/K1/K1_22dof.urdf"


def k1_urdf_path() -> str:
    """Resolve the K1 URDF without silently substituting another robot asset."""

    configured = os.environ.get("AGILE_K1_URDF_PATH")
    if configured:
        return str(Path(configured).expanduser().resolve())
    return str(_DEFAULT_K1_URDF_PATH)


def k1_usd_cache_dir() -> str:
    configured = os.environ.get("AGILE_K1_USD_CACHE_DIR", "~/.cache/agile/k1-usd")
    return str(Path(configured).expanduser().resolve())


@dataclass(frozen=True)
class _Motor:
    effort: float
    velocity: float
    knee_velocity: float
    armature: float
    natural_frequency_hz: float = 4.0
    damping_ratio: float = 1.5

    @property
    def stiffness(self) -> float:
        return self.armature * (2.0 * pi * self.natural_frequency_hz) ** 2

    @property
    def damping(self) -> float:
        return 2.0 * self.damping_ratio * self.armature * (2.0 * pi * self.natural_frequency_hz)

    @property
    def action_scale(self) -> float:
        """Joint offset that requests 25% of the zero-speed effort limit.

        A uniform position scale is not compatible with the motor-specific
        stiffness values below.  Normalizing by stiffness gives every joint
        the same physically meaningful exploration authority while preserving
        the measured torque limits.
        """

        return 0.25 * self.effort / self.stiffness


_LEG_MOTORS = {
    ".*_Hip_Pitch": _Motor(68.0, 14.66, 1.88, 0.0478125),
    ".*_Hip_Roll": _Motor(76.0, 12.57, 2.62, 0.0339552),
    ".*_Hip_Yaw": _Motor(38.3, 17.59, 7.85, 0.0282528),
    ".*_Knee_Pitch": _Motor(112.0, 12.57, 2.09, 0.095625, damping_ratio=1.0),
    ".*_Ankle_Pitch": _Motor(38.3, 17.59, 7.85, 0.0565056),
    ".*_Ankle_Roll": _Motor(38.3, 17.59, 7.85, 0.0565056),
}

K1_LOCOMOTION_ACTION_SCALE = {pattern: motor.action_scale for pattern, motor in _LEG_MOTORS.items()}
"""Per-joint target offsets normalized to 25% of each motor's effort limit."""


def _leg_parameter(name: str) -> dict[str, float]:
    return {pattern: getattr(model, name) for pattern, model in _LEG_MOTORS.items()}


K1_DELAYED_PD_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        asset_path=k1_urdf_path(),
        usd_dir=k1_usd_cache_dir(),
        usd_file_name="K1_22dof.usd",
        activate_contact_sensors=True,
        replace_cylinders_with_capsules=False,
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, DEFAULT_TRUNK_HEIGHT),
        joint_pos={
            "Left_Shoulder_Roll": -1.3,
            "Right_Shoulder_Roll": 1.3,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": BoosterDelayedPDActuatorCfg(
            joint_names_expr=list(_LEG_MOTORS),
            min_delay=MIN_DELAY_STEPS,
            max_delay=MAX_DELAY_STEPS,
            effort_limit_sim=_leg_parameter("effort"),
            velocity_limit_sim=_leg_parameter("velocity"),
            knee_point_velocity=_leg_parameter("knee_velocity"),
            armature=_leg_parameter("armature"),
            stiffness=_leg_parameter("stiffness"),
            damping=_leg_parameter("damping"),
        ),
        "arms": BoosterDelayedPDActuatorCfg(
            joint_names_expr=ARM_JOINT_NAMES,
            min_delay=MIN_DELAY_STEPS,
            max_delay=MAX_DELAY_STEPS,
            effort_limit_sim=14.0,
            velocity_limit_sim=33.51,
            knee_point_velocity=5.24,
            armature=0.001,
            stiffness=0.001 * (2.0 * pi * 10.0) ** 2,
            damping=2.0 * 2.0 * 0.001 * (2.0 * pi * 10.0),
        ),
        "head": BoosterDelayedPDActuatorCfg(
            joint_names_expr=HEAD_JOINT_NAMES,
            min_delay=MIN_DELAY_STEPS,
            max_delay=MAX_DELAY_STEPS,
            effort_limit_sim=6.0,
            velocity_limit_sim=7.85,
            knee_point_velocity=7.85,
            armature=0.001,
            stiffness=0.001 * (2.0 * pi * 10.0) ** 2,
            damping=2.0 * 2.0 * 0.001 * (2.0 * pi * 10.0),
        ),
    },
)


K1_LOCOMOTION_CFG = K1_DELAYED_PD_CFG.replace(
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, DEFAULT_TRUNK_HEIGHT),
        joint_pos={
            "ALeft_Shoulder_Pitch": 0.2,
            "Left_Shoulder_Roll": -1.25,
            "Left_Elbow_Yaw": -0.5,
            "ARight_Shoulder_Pitch": 0.2,
            "Right_Shoulder_Roll": 1.25,
            "Right_Elbow_Yaw": 0.5,
            "Left_Hip_Pitch": -0.15,
            "Left_Knee_Pitch": 0.3,
            "Left_Ankle_Pitch": -0.15,
            "Right_Hip_Pitch": -0.15,
            "Right_Knee_Pitch": 0.3,
            "Right_Ankle_Pitch": -0.15,
        },
        joint_vel={".*": 0.0},
    )
)
"""K1 articulation with the deployed LOCO neutral posture as its action offset."""
