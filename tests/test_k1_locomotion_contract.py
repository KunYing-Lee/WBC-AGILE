# SPDX-FileCopyrightText: Copyright (c) 2026 KunYing Lee
# SPDX-License-Identifier: Apache-2.0

"""CPU-only source-contract tests for Booster K1 locomotion."""

from __future__ import annotations

import ast
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
TASK_PATH = ROOT / "agile/rl_env/tasks/locomotion/k1/velocity_env_cfg.py"
REGISTER_PATH = ROOT / "agile/rl_env/tasks/locomotion/k1/__init__.py"
ROBOT_PATH = ROOT / "agile/rl_env/assets/robots/booster_k1.py"
COMMAND_SCHEDULE_PATH = ROOT / "agile/sim2mujoco/configs/k1_command_bounds.yaml"
AXIS_EVAL_PATH = ROOT / "agile/algorithms/evaluation/configs/k1_velocity_axes_v1.yaml"
SEQUENCE_EVAL_PATH = ROOT / "agile/algorithms/evaluation/configs/k1_velocity_sequence_v1.yaml"
CURRICULUM_PATH = ROOT / "agile/rl_env/mdp/curriculums/task_curriculum.py"
CHECKPOINT_STATE_PATH = ROOT / "agile/rl_env/rsl_rl/checkpoint_state.py"
VECENV_WRAPPER_PATH = ROOT / "agile/rl_env/rsl_rl/vecenv_wrapper.py"
TRAIN_PATH = ROOT / "scripts/train.py"
EVAL_PATH = ROOT / "scripts/eval.py"


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _class(tree: ast.Module, name: str) -> ast.ClassDef:
    return next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name)


def _call_assignment(node: ast.ClassDef, name: str) -> ast.Call:
    assignment = next(
        item
        for item in node.body
        if isinstance(item, ast.Assign)
        and len(item.targets) == 1
        and isinstance(item.targets[0], ast.Name)
        and item.targets[0].id == name
    )
    assert isinstance(assignment.value, ast.Call)
    return assignment.value


def _keyword(call: ast.Call, name: str) -> ast.expr:
    return next(keyword.value for keyword in call.keywords if keyword.arg == name)


def _top_level_literals(tree: ast.Module) -> dict[str, object]:
    values: dict[str, object] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                values[node.targets[0].id] = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                pass
    return values


def test_k1_velocity_command_bounds() -> None:
    values = _top_level_literals(_tree(TASK_PATH))
    assert values["K1_LIN_VEL_X_RANGE"] == (-1.0, 1.5)
    assert values["K1_LIN_VEL_Y_RANGE"] == (-1.5, 1.5)
    assert values["K1_ANG_VEL_Z_RANGE"] == (-2.0, 2.0)
    assert values["K1_INITIAL_LIN_VEL_X_RANGE"] == (-0.25, 0.35)
    assert values["K1_INITIAL_LIN_VEL_Y_RANGE"] == (-0.25, 0.25)
    assert values["K1_INITIAL_ANG_VEL_Z_RANGE"] == (-0.5, 0.5)
    assert values["K1_COMMAND_MODE_WEIGHTS"] == (0.30, 0.25, 0.20, 0.25)
    assert values["K1_COMMAND_MIN_MAGNITUDES"] == (0.10, 0.10, 0.20)


def test_k1_velocity_command_curriculum_expands_to_full_contract() -> None:
    curriculum = _class(_tree(TASK_PATH), "CurriculumCfg")
    term = _call_assignment(curriculum, "velocity_command_ranges")
    assert ast.unparse(_keyword(term, "func")) == "mdp.velocity_command_range_success"

    params = _keyword(term, "params")
    assert isinstance(params, ast.Dict)
    entries = {
        ast.literal_eval(key): value for key, value in zip(params.keys, params.values, strict=True) if key is not None
    }
    assert ast.literal_eval(entries["command_name"]) == "base_velocity"
    assert ast.literal_eval(entries["planar_error_threshold"]) == 0.18
    assert ast.literal_eval(entries["yaw_error_threshold"]) == 0.25
    assert ast.literal_eval(entries["minimum_episode_age_ratio"]) == 0.30
    assert ast.literal_eval(entries["hold_steps"]) == 2_500
    assert ast.literal_eval(entries["scale_increment"]) == 0.10
    assert ast.unparse(entries["start_ranges"]) == (
        "{'lin_vel_x': K1_INITIAL_LIN_VEL_X_RANGE, 'lin_vel_y': K1_INITIAL_LIN_VEL_Y_RANGE, "
        "'ang_vel_z': K1_INITIAL_ANG_VEL_Z_RANGE}"
    )
    assert ast.unparse(entries["terminal_ranges"]) == (
        "{'lin_vel_x': K1_LIN_VEL_X_RANGE, 'lin_vel_y': K1_LIN_VEL_Y_RANGE, 'ang_vel_z': K1_ANG_VEL_Z_RANGE}"
    )


def test_k1_velocity_curriculum_is_distributed_and_checkpointed() -> None:
    curriculum = _class(_tree(CURRICULUM_PATH), "velocity_command_range_success")
    source = ast.unparse(curriculum)
    assert "checkpoint_state_required = True" in source
    assert "def freeze_checkpoint_state" in source
    assert "if self._checkpoint_state_frozen" in source
    assert "def checkpoint_state_dict" in source
    assert "def load_checkpoint_state_dict" in source
    assert "def synchronize_checkpoint_state" in source
    assert "torch.distributed.all_reduce(summed, op=torch.distributed.ReduceOp.SUM)" in source
    assert "torch.distributed.all_reduce(minimum, op=torch.distributed.ReduceOp.MIN)" in source
    assert "if not distributed and self._successful_steps >= hold_steps" in source
    assert "self._update_ranges(env)" in source

    checkpoint_runner = _class(_tree(CHECKPOINT_STATE_PATH), "EnvironmentStateOnPolicyRunner")
    runner_source = ast.unparse(checkpoint_runner)
    assert "collect_environment_state(self.env)" in runner_source
    assert "restore_environment_state(" in runner_source
    assert "require_topology_match=self._require_environment_topology_match" in runner_source
    assert "configure_synchronized_step_callback" in runner_source

    wrapper = _class(_tree(VECENV_WRAPPER_PATH), "RslRlVecEnvWrapper")
    wrapper_source = ast.unparse(wrapper)
    assert "def configure_synchronized_step_callback" in wrapper_source
    assert "self._synchronized_step_callback()" in wrapper_source

    train_source = ast.unparse(_tree(TRAIN_PATH))
    assert "runner = EnvironmentStateOnPolicyRunner" in train_source


def test_k1_evaluation_restores_checkpoint_curriculum_without_training_topology() -> None:
    source = ast.unparse(_tree(EVAL_PATH))
    assert "ppo_runner = EnvironmentStateOnPolicyRunner" in source
    assert "require_environment_topology_match=False" in source
    assert "freeze_environment_state_after_load=True" in source
    assert "ppo_runner.load(resume_path, load_optimizer=False)" in source
    assert "checkpointed_curriculum_cfg = _checkpointed_curriculum_cfg(env_cfg)" in source
    assert "env_cfg.curriculum = checkpointed_curriculum_cfg" in source

    restore = next(
        node
        for node in _tree(CHECKPOINT_STATE_PATH).body
        if isinstance(node, ast.FunctionDef) and node.name == "restore_environment_state"
    )
    restore_source = ast.unparse(restore)
    assert "require_topology_match: bool=True" in restore_source
    assert "if require_topology_match and saved_contract != expected_contract" in restore_source


def test_k1_locomotion_controls_exact_leg_contract() -> None:
    actions = _class(_tree(TASK_PATH), "ActionsCfg")
    action = _call_assignment(actions, "joint_pos")
    assert ast.unparse(_keyword(action, "joint_names")) == "booster_k1.K1_LEG_JOINT_NAMES"
    assert ast.unparse(_keyword(action, "scale")) == "booster_k1.K1_LOCOMOTION_ACTION_SCALE"
    assert ast.literal_eval(_keyword(action, "preserve_order")) is True


def test_k1_training_holds_upper_body_and_stratifies_commands() -> None:
    tree = _tree(TASK_PATH)
    actions = _class(tree, "ActionsCfg")
    action_source = ast.unparse(actions)
    assert "random_pos = None" in action_source
    assert "upper_body_hold = mdp.HoldJointPositionActionCfg" in action_source
    assert "joint_names=booster_k1.K1_HEAD_JOINT_NAMES + booster_k1.K1_ARM_JOINT_NAMES" in action_source

    commands = _class(tree, "CommandsCfg")
    command = _call_assignment(commands, "base_velocity")
    assert ast.unparse(command.func) == "mdp.StratifiedUniformVelocityCommandCfg"
    assert ast.unparse(_keyword(command, "mode_weights")) == "K1_COMMAND_MODE_WEIGHTS"
    assert ast.literal_eval(_keyword(command, "rel_standing_envs")) == 0.10


def test_k1_nominal_first_disables_dr_and_uses_dense_rewards() -> None:
    tree = _tree(TASK_PATH)
    events = _class(tree, "NominalLocomotionEventCfg")
    event_source = ast.unparse(events)
    for name in (
        "randomize_physics_material",
        "randomize_actuator_gains",
        "randomize_base_mass",
        "randomize_base_com",
        "apply_external_force_torque",
        "push_robot",
    ):
        assert f"{name} = None" in event_source

    rewards = _class(tree, "RewardsCfg")
    linear_tracking = _call_assignment(rewards, "track_lin_vel_xy_exp")
    linear_params = ast.literal_eval(_keyword(linear_tracking, "params"))
    assert ast.literal_eval(_keyword(linear_tracking, "weight")) == 5.0
    assert linear_params["std"] == 0.2
    yaw_tracking = _call_assignment(rewards, "track_ang_vel")
    yaw_params_node = _keyword(yaw_tracking, "params")
    assert isinstance(yaw_params_node, ast.Dict)
    yaw_params = {
        ast.literal_eval(key): value
        for key, value in zip(yaw_params_node.keys, yaw_params_node.values, strict=True)
        if key is not None
    }
    assert ast.literal_eval(_keyword(yaw_tracking, "weight")) == 5.0
    assert ast.literal_eval(yaw_params["std"]) == 0.2
    action_rate = _call_assignment(rewards, "action_rate")
    assert ast.literal_eval(_keyword(action_rate, "weight")) == -0.005
    _call_assignment(rewards, "feet_air_time")

    env = _class(tree, "K1LowerVelocityEnvCfg")
    post_init = next(node for node in env.body if isinstance(node, ast.FunctionDef) and node.name == "__post_init__")
    source = ast.unparse(post_init)
    assert "self.scene.terrain.terrain_type = 'plane'" in source
    assert "self.scene.terrain.terrain_generator = None" in source


def test_k1_locomotion_uses_deployed_neutral_leg_posture() -> None:
    robot_tree = _tree(ROBOT_PATH)
    assignment = next(
        node
        for node in robot_tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "K1_LOCOMOTION_CFG"
    )
    assert isinstance(assignment.value, ast.Call)
    init_state = _keyword(assignment.value, "init_state")
    assert isinstance(init_state, ast.Call)
    joint_pos = ast.literal_eval(_keyword(init_state, "joint_pos"))
    assert joint_pos["Left_Hip_Pitch"] == -0.15
    assert joint_pos["Left_Knee_Pitch"] == 0.3
    assert joint_pos["Left_Ankle_Pitch"] == -0.15
    assert joint_pos["Right_Hip_Pitch"] == -0.15
    assert joint_pos["Right_Knee_Pitch"] == 0.3
    assert joint_pos["Right_Ankle_Pitch"] == -0.15


def test_k1_locomotion_rates_match_wbc_agile_contract() -> None:
    env = _class(_tree(TASK_PATH), "K1LowerVelocityEnvCfg")
    post_init = next(node for node in env.body if isinstance(node, ast.FunctionDef) and node.name == "__post_init__")
    wanted = {"self.controller_freq", "self.physics_freq"}
    assignments = {
        ast.unparse(node.targets[0]): ast.literal_eval(node.value)
        for node in post_init.body
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and ast.unparse(node.targets[0]) in wanted
    }
    assert assignments["self.controller_freq"] == 50.0
    assert assignments["self.physics_freq"] == 200.0


def test_velocity_k1_task_is_registered() -> None:
    calls = [node for node in ast.walk(_tree(REGISTER_PATH)) if isinstance(node, ast.Call)]
    register = next(call for call in calls if ast.unparse(call.func) == "gym.register")
    assert ast.literal_eval(_keyword(register, "id")) == "Velocity-K1-v0"
    kwargs = _keyword(register, "kwargs")
    assert isinstance(kwargs, ast.Dict)
    entries = {
        ast.literal_eval(key): ast.unparse(value)
        for key, value in zip(kwargs.keys, kwargs.values, strict=True)
        if key is not None
    }
    assert "K1LowerVelocityEnvCfg" in entries["env_cfg_entry_point"]
    assert "K1VelocityPpoRunnerCfg" in entries["rsl_rl_cfg_entry_point"]


def test_k1_sim2mujoco_schedules_cover_command_box() -> None:
    schedules = yaml.safe_load(COMMAND_SCHEDULE_PATH.read_text(encoding="utf-8"))
    commands = {
        (float(entry[1]), float(entry[2]), float(entry[3])) for entries in schedules.values() for entry in entries
    }

    assert {command[0] for command in commands} >= {-1.0, 1.5}
    assert {command[1] for command in commands} >= {-1.5, 1.5}
    assert {command[2] for command in commands} >= {-2.0, 2.0}

    expected_corners = {(vx, vy, wz) for vx in (-1.0, 1.5) for vy in (-1.5, 1.5) for wz in (-2.0, 2.0)}
    assert commands >= expected_corners


def test_k1_nominal_eval_covers_stand_and_each_axis_direction() -> None:
    evaluation = yaml.safe_load(AXIS_EVAL_PATH.read_text(encoding="utf-8"))["evaluation"]
    assert evaluation["task_name"] == "Velocity-K1-v0"
    assert evaluation["num_envs"] == 7
    assert evaluation["env_overrides"]["events"]["disable_all"] is True

    commands = {
        tuple(environment["schedule"][0]["commands"]["base_velocity"].values())
        for environment in evaluation["environments"]
    }
    assert commands == {
        (0.0, 0.0, 0.0),
        (0.5, 0.0, 0.0),
        (-0.5, 0.0, 0.0),
        (0.0, 0.5, 0.0),
        (0.0, -0.5, 0.0),
        (0.0, 0.0, 0.8),
        (0.0, 0.0, -0.8),
    }


def test_k1_video_sequence_is_time_ordered() -> None:
    evaluation = yaml.safe_load(SEQUENCE_EVAL_PATH.read_text(encoding="utf-8"))["evaluation"]
    schedule = evaluation["environments"][0]["schedule"]
    assert [step["time"] for step in schedule] == [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0]


def test_k1_eval_adds_nominal_plane_and_evaluation_observations() -> None:
    env = _class(_tree(TASK_PATH), "K1LowerVelocityEnvCfg")
    eval_method = next(node for node in env.body if isinstance(node, ast.FunctionDef) and node.name == "eval")
    source = ast.unparse(eval_method)
    assert "self.scene.terrain.terrain_type = 'plane'" in source
    assert "self.scene.terrain.terrain_generator = None" in source
    assert "self.actions.random_pos = None" in source
    assert "self.actions.upper_body_hold = mdp.HoldJointPositionActionCfg" in source
    assert "joint_names=booster_k1.K1_HEAD_JOINT_NAMES + booster_k1.K1_ARM_JOINT_NAMES" in source
    assert "self.observations.eval = mdp.EvaluationObservationsCfg()" in source
