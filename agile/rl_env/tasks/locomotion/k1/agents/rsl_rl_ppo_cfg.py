# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math

from isaaclab.utils import configclass

from agile.rl_env.mdp.symmetry.symmetry_k1 import lr_mirror_K1  # noqa: F401
from agile.rl_env.rsl_rl import (  # noqa: F401
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
    RslRlSymmetryCfg,
)


@configclass
class K1VelocityPpoRunnerCfg(RslRlOnPolicyRunnerCfg):
    seed = 42
    num_steps_per_env = 24
    max_iterations = 100_000
    save_interval = 250
    experiment_name = "velocity_k1_lower"
    run_name = "velocity_k1_lower"
    wandb_project = "Velocity-K1-Lower"
    empirical_normalization = False
    enable_entropy_coef_annealing = True
    entropy_coef_annealing_start_progress = 0.3
    enable_entropy_coef_annealing_success_rate = 0.8
    entropy_annealing_decay_rate = 0.9995
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.5,
        actor_hidden_dims=[256, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation=True,
            use_mirror_loss=False,
            data_augmentation_func=lr_mirror_K1,
        ),
    )


@configclass
class K1VelocityStrideFinetunePpoRunnerCfg(RslRlOnPolicyRunnerCfg):
    """Actor-only model-5750 finetune with bounded exploration and behavior anchoring."""

    seed = 42
    num_steps_per_env = 24
    max_iterations = 10_000
    save_interval = 100
    experiment_name = "velocity_k1_stride_finetune"
    run_name = "model5750_stride_v1"
    wandb_project = "Velocity-K1-Lower"
    empirical_normalization = False
    enable_entropy_coef_annealing = True
    entropy_coef_annealing_start_progress = 0.0
    enable_entropy_coef_annealing_success_rate = 0.8
    entropy_annealing_decay_rate = 0.9995
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.35,
        noise_std_type="log",
        log_std_range=(math.log(0.05), math.log(0.60)),
        actor_hidden_dims=[256, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.001,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        reference_policy_kl_coef=0.1,
        reference_policy_kl_cvar_fraction=1.0,
        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation=True,
            use_mirror_loss=False,
            data_augmentation_func=lr_mirror_K1,
        ),
    )


@configclass
class K1VelocityStrideBalancedFinetunePpoRunnerCfg(K1VelocityStrideFinetunePpoRunnerCfg):
    """Direction-balanced successor to the fixed-threshold stride finetune."""

    experiment_name = "velocity_k1_stride_balanced_finetune"
    run_name = "model5750_stride_balanced_v2"

    # Actor-only warm starts intentionally discard the parent's critic. With
    # five learning epochs and four mini-batches, 2,000 optimizer steps equal
    # 100 on-policy iterations of critic-only fitting; the PPO implementation
    # then ramps actor losses in over the following 100 iterations.
    algorithm = K1VelocityStrideFinetunePpoRunnerCfg.algorithm.replace(critic_warmup_steps=2_000)
