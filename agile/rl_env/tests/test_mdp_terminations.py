# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest
from unittest.mock import MagicMock

import torch

from agile.rl_env.tests.utils import APP_IS_READY

if APP_IS_READY:
    from isaaclab.managers import SceneEntityCfg

    from agile.rl_env.mdp.terminations import stable_upright


@unittest.skipUnless(APP_IS_READY, "Isaac Lab app is required")
class TestStableUprightTermination(unittest.TestCase):
    def setUp(self) -> None:
        self.num_envs = 4
        self.env = MagicMock()
        self.env.num_envs = self.num_envs
        self.env.device = "cpu"
        self.env.step_dt = 0.02

        self.robot = MagicMock()
        self.robot.data.root_pos_w = torch.tensor([[0.0, 0.0, 0.54]] * self.num_envs)
        self.robot.data.projected_gravity_b = torch.tensor(
            [
                [0.0, 0.0, -1.0],  # valid upright state
                [-1.0, 0.0, 0.0],  # bridge pose: height is valid but trunk is horizontal
                [0.0, 0.0, -1.0],  # one foot is not supporting
                [0.0, 0.0, -1.0],  # base is moving too quickly
            ]
        )
        self.robot.data.root_lin_vel_b = torch.zeros((self.num_envs, 3))
        self.robot.data.root_lin_vel_b[3, 0] = 0.5
        self.robot.data.root_ang_vel_b = torch.zeros((self.num_envs, 3))

        self.height_sensor = MagicMock()
        self.height_sensor.data.ray_hits_w = torch.zeros((self.num_envs, 1, 3))

        self.contact_sensor = MagicMock()
        self.contact_sensor.data.net_forces_w = torch.zeros((self.num_envs, 4, 3))
        self.contact_sensor.data.net_forces_w[:, 0:2, 2] = 100.0
        self.contact_sensor.data.net_forces_w[2, 1, 2] = 0.0
        self.contact_sensor.data.net_forces_w[1, 2, 2] = 100.0

        scene_items = {"robot": self.robot}
        self.env.scene.__getitem__ = lambda _scene, name: scene_items[name]
        self.env.scene.sensors = {
            "height": self.height_sensor,
            "contacts": self.contact_sensor,
        }

        self.asset_cfg = SceneEntityCfg("robot")
        self.height_cfg = SceneEntityCfg("height")
        self.feet_cfg = SceneEntityCfg("contacts")
        self.feet_cfg.body_ids = [0, 1]
        self.undesired_cfg = SceneEntityCfg("contacts")
        self.undesired_cfg.body_ids = [2, 3]

        # Bypass ManagerTermBase construction: this unit test supplies the fully
        # resolved scene/config state consumed by the term itself.
        self.term = object.__new__(stable_upright)
        self.term._env = self.env
        self.term.stable_steps = torch.zeros(self.num_envs, dtype=torch.int64)

    def evaluate(self, duration_s: float = 1.0) -> torch.Tensor:
        return self.term(
            self.env,
            asset_cfg=self.asset_cfg,
            height_sensor_cfg=self.height_cfg,
            feet_sensor_cfg=self.feet_cfg,
            undesired_contact_sensor_cfg=self.undesired_cfg,
            min_height=0.52,
            max_tilt_angle_rad=0.2617993877991494,
            max_lin_vel=0.25,
            max_ang_vel=0.5,
            min_foot_contact_force=5.0,
            max_undesired_contact_force=5.0,
            duration_s=duration_s,
        )

    def test_requires_every_condition_for_full_duration(self) -> None:
        for _ in range(49):
            self.assertFalse(torch.any(self.evaluate()))

        result = self.evaluate()
        self.assertTrue(result[0])
        self.assertFalse(torch.any(result[1:]))

    def test_invalid_step_resets_continuous_hold(self) -> None:
        for _ in range(25):
            self.evaluate()

        self.robot.data.root_ang_vel_b[0, 0] = 1.0
        self.evaluate()
        self.robot.data.root_ang_vel_b[0, 0] = 0.0

        for _ in range(49):
            self.assertFalse(self.evaluate()[0])
        self.assertTrue(self.evaluate()[0])


if __name__ == "__main__":
    unittest.main()
