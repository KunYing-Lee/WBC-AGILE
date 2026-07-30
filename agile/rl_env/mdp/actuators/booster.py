# SPDX-FileCopyrightText: Copyright (c) 2026 KunYing Lee
# SPDX-License-Identifier: Apache-2.0

"""Booster actuator models with an explicit torque-speed envelope."""

from __future__ import annotations

from typing import Any

import torch

from isaaclab.actuators import DelayedPDActuator, DelayedPDActuatorCfg
from isaaclab.utils import configclass
from isaaclab.utils.types import ArticulationActions


class BoosterDelayedPDActuator(DelayedPDActuator):
    """Delayed PD actuator with Booster's piecewise-linear torque-speed curve."""

    cfg: BoosterDelayedPDActuatorCfg

    def __init__(self, cfg: BoosterDelayedPDActuatorCfg, *args: Any, **kwargs: Any) -> None:
        super().__init__(cfg, *args, **kwargs)
        self.knee_point_velocity = self._parse_joint_parameter(cfg.knee_point_velocity, self.velocity_limit)
        self.knee_point_velocity = torch.clamp(self.knee_point_velocity, min=0.0)
        self.knee_point_velocity = torch.minimum(self.knee_point_velocity, self.velocity_limit)
        self._joint_vel = torch.zeros_like(self.computed_effort)
        self._torque_speed_denominator = (self.velocity_limit - self.knee_point_velocity).clamp(min=1.0e-6)

    def compute(
        self,
        control_action: ArticulationActions,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
    ) -> ArticulationActions:
        self._joint_vel[:] = joint_vel
        return super().compute(control_action, joint_pos, joint_vel)

    def _clip_effort(self, effort: torch.Tensor) -> torch.Tensor:
        joint_speed = self._joint_vel.abs()
        linear_limit = self.effort_limit * (self.velocity_limit - joint_speed) / self._torque_speed_denominator
        max_effort = linear_limit.clamp(min=0.0)
        max_effort = torch.minimum(max_effort, self.effort_limit)
        max_effort = torch.where(~torch.isfinite(self.velocity_limit), self.effort_limit, max_effort)
        max_effort = torch.where(self.velocity_limit <= 0.0, torch.zeros_like(max_effort), max_effort)
        return torch.clip(effort, min=-max_effort, max=max_effort)


@configclass
class BoosterDelayedPDActuatorCfg(DelayedPDActuatorCfg):
    """Configuration for :class:`BoosterDelayedPDActuator`."""

    class_type: type = BoosterDelayedPDActuator
    knee_point_velocity: dict[str, float] | float | None = None
