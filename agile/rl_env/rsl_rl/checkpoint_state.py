# SPDX-FileCopyrightText: Copyright (c) 2026 KunYing Lee
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed environment state support for RSL-RL checkpoints."""

from __future__ import annotations

from typing import Any

import torch
from rsl_rl.runners import OnPolicyRunner

_ENVIRONMENT_STATE_KEY = "environment_state"
_SCHEMA_VERSION = 1


def _distributed_world_size() -> int:
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return 1
    return torch.distributed.get_world_size()


def _required_curriculum_terms(env: Any) -> dict[str, Any]:
    unwrapped = env.unwrapped
    manager = getattr(unwrapped, "curriculum_manager", None)
    if manager is None:
        return {}

    terms: dict[str, Any] = {}
    for name, cfg in zip(manager._term_names, manager._term_cfgs, strict=True):
        term = cfg.func
        if not getattr(term, "checkpoint_state_required", False):
            continue
        if not callable(getattr(term, "checkpoint_state_dict", None)):
            raise TypeError(f"Required curriculum term '{name}' has no checkpoint_state_dict().")
        if not callable(getattr(term, "load_checkpoint_state_dict", None)):
            raise TypeError(f"Required curriculum term '{name}' has no load_checkpoint_state_dict().")
        if not callable(getattr(term, "synchronize_checkpoint_state", None)):
            raise TypeError(f"Required curriculum term '{name}' has no synchronize_checkpoint_state().")
        terms[name] = term
    return terms


def synchronize_environment_state(env: Any) -> None:
    """Synchronize required state at a step boundary shared by every rank."""
    unwrapped = env.unwrapped
    for name, term in sorted(_required_curriculum_terms(env).items()):
        term.synchronize_checkpoint_state(unwrapped)
        unwrapped.curriculum_manager._curriculum_state[name] = term.checkpoint_state_dict()["scale"]


def collect_environment_state(env: Any) -> dict[str, Any] | None:
    """Collect state for every curriculum term that declares it mandatory."""
    terms = _required_curriculum_terms(env)
    if not terms:
        return None
    return {
        "schema_version": _SCHEMA_VERSION,
        "contract": {
            "distributed_world_size": _distributed_world_size(),
            "environments_per_rank": int(env.num_envs),
        },
        "curriculum_terms": {
            name: {
                "class": f"{type(term).__module__}.{type(term).__qualname__}",
                "state": term.checkpoint_state_dict(),
            }
            for name, term in sorted(terms.items())
        },
    }


def restore_environment_state(env: Any, payload: dict[str, Any] | None) -> None:
    """Restore required curriculum state, rejecting incomplete or incompatible payloads."""
    terms = _required_curriculum_terms(env)
    if not terms:
        return
    if payload is None:
        raise RuntimeError("Checkpoint is missing required environment_state; exact training resume is impossible.")
    if (
        set(payload) != {"schema_version", "contract", "curriculum_terms"}
        or payload["schema_version"] != _SCHEMA_VERSION
    ):
        raise RuntimeError("Checkpoint environment_state schema is missing or unsupported.")
    expected_contract = {
        "distributed_world_size": _distributed_world_size(),
        "environments_per_rank": int(env.num_envs),
    }
    if payload["contract"] != expected_contract:
        raise RuntimeError(
            f"Checkpoint environment contract {payload['contract']} does not match current {expected_contract}."
        )

    saved_terms = payload["curriculum_terms"]
    if not isinstance(saved_terms, dict) or set(saved_terms) != set(terms):
        raise RuntimeError(
            f"Checkpoint curriculum terms {sorted(saved_terms) if isinstance(saved_terms, dict) else saved_terms} "
            f"do not match required terms {sorted(terms)}."
        )
    unwrapped = env.unwrapped
    for name, term in sorted(terms.items()):
        entry = saved_terms[name]
        expected_class = f"{type(term).__module__}.{type(term).__qualname__}"
        if not isinstance(entry, dict) or set(entry) != {"class", "state"} or entry["class"] != expected_class:
            raise RuntimeError(f"Checkpoint contract mismatch for curriculum term '{name}'.")
        term.load_checkpoint_state_dict(entry["state"], unwrapped)
        unwrapped.curriculum_manager._curriculum_state[name] = term.checkpoint_state_dict()["scale"]


class EnvironmentStateOnPolicyRunner(OnPolicyRunner):
    """OnPolicyRunner that checkpoints all mandatory mutable environment state."""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        if self.is_distributed and _required_curriculum_terms(self.env):
            self.env.configure_synchronized_step_callback(
                interval=self.num_steps_per_env,
                callback=lambda: synchronize_environment_state(self.env),
            )
            synchronize_environment_state(self.env)

    def save(self, path: str, infos: dict[str, Any] | None = None) -> None:
        environment_state = collect_environment_state(self.env)
        checkpoint_infos = dict(infos or {})
        if environment_state is not None:
            checkpoint_infos[_ENVIRONMENT_STATE_KEY] = environment_state
        super().save(path, checkpoint_infos or None)

    def load(self, path: str, load_optimizer: bool = True) -> dict[str, Any] | None:
        infos = super().load(path, load_optimizer=load_optimizer)
        payload = infos.get(_ENVIRONMENT_STATE_KEY) if isinstance(infos, dict) else None
        restore_environment_state(self.env, payload)
        return infos
