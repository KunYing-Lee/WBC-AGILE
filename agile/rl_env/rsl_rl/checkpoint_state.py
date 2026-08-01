# SPDX-FileCopyrightText: Copyright (c) 2026 KunYing Lee
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed environment state support for RSL-RL checkpoints."""

from __future__ import annotations

import hashlib
import re
from typing import Any

import torch
from rsl_rl.runners import OnPolicyRunner

_ENVIRONMENT_STATE_KEY = "environment_state"
_WARM_START_KEY = "actor_only_warm_start"
_SCHEMA_VERSION = 1
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def checkpoint_sha256(path: str) -> str:
    """Return the SHA256 of a checkpoint without loading it."""
    digest = hashlib.sha256()
    with open(path, "rb") as checkpoint_file:
        for chunk in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
        if not callable(getattr(term, "freeze_checkpoint_state", None)):
            raise TypeError(f"Required curriculum term '{name}' has no freeze_checkpoint_state().")
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


def freeze_environment_state(env: Any) -> None:
    """Freeze every restored mandatory curriculum term for deterministic evaluation."""
    for term in _required_curriculum_terms(env).values():
        term.freeze_checkpoint_state()


def restore_environment_state(
    env: Any,
    payload: dict[str, Any] | None,
    *,
    require_topology_match: bool = True,
) -> None:
    """Restore required curriculum state, rejecting incomplete or incompatible payloads.

    Training resume requires the original distributed topology because optimizer
    continuation must be exact. Evaluation deliberately uses a different number
    of environments, but still validates the saved topology contract before
    restoring the checkpoint's curriculum ranges and mutable state.
    """
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
    saved_contract = payload["contract"]
    if (
        not isinstance(saved_contract, dict)
        or set(saved_contract) != {"distributed_world_size", "environments_per_rank"}
        or not isinstance(saved_contract["distributed_world_size"], int)
        or not isinstance(saved_contract["environments_per_rank"], int)
        or saved_contract["distributed_world_size"] <= 0
        or saved_contract["environments_per_rank"] <= 0
    ):
        raise RuntimeError(f"Checkpoint environment contract is invalid: {saved_contract}.")

    expected_contract = {
        "distributed_world_size": _distributed_world_size(),
        "environments_per_rank": int(env.num_envs),
    }
    if require_topology_match and saved_contract != expected_contract:
        raise RuntimeError(
            f"Checkpoint environment contract {saved_contract} does not match current {expected_contract}."
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

    def __init__(
        self,
        *args: Any,
        require_environment_topology_match: bool = True,
        freeze_environment_state_after_load: bool = False,
        **kwargs: Any,
    ):
        self._require_environment_topology_match = require_environment_topology_match
        self._freeze_environment_state_after_load = freeze_environment_state_after_load
        self._warm_start_provenance: dict[str, Any] | None = None
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
        if self._warm_start_provenance is not None:
            checkpoint_infos[_WARM_START_KEY] = self._warm_start_provenance
        if environment_state is not None:
            checkpoint_infos[_ENVIRONMENT_STATE_KEY] = environment_state
        super().save(path, checkpoint_infos or None)

    def load(self, path: str, load_optimizer: bool = True) -> dict[str, Any] | None:
        infos = super().load(path, load_optimizer=load_optimizer)
        self._warm_start_provenance = infos.get(_WARM_START_KEY) if isinstance(infos, dict) else None
        payload = infos.get(_ENVIRONMENT_STATE_KEY) if isinstance(infos, dict) else None
        restore_environment_state(
            self.env,
            payload,
            require_topology_match=self._require_environment_topology_match,
        )
        if self._freeze_environment_state_after_load:
            freeze_environment_state(self.env)
        return infos

    def load_actor_only(self, path: str, expected_sha256: str) -> dict[str, Any]:
        """Load only actor weights and leave all training state freshly initialized.

        This is deliberately separate from :meth:`load`: optimizer, critic,
        exploration parameters, iteration counters, curriculum state, and
        rollout storage are never imported from the parent checkpoint.
        """
        expected_sha256 = expected_sha256.lower()
        if _SHA256_PATTERN.fullmatch(expected_sha256) is None:
            raise ValueError("Expected checkpoint SHA256 must be exactly 64 lowercase hexadecimal characters.")
        actual_sha256 = checkpoint_sha256(path)
        if actual_sha256 != expected_sha256:
            raise RuntimeError(
                f"Parent checkpoint SHA256 mismatch: expected {expected_sha256}, got {actual_sha256}."
            )

        loaded_dict = torch.load(path, map_location=self.device, weights_only=False)
        if not isinstance(loaded_dict, dict) or not isinstance(loaded_dict.get("model_state_dict"), dict):
            raise RuntimeError("Parent checkpoint is missing model_state_dict.")

        source_state = loaded_dict["model_state_dict"]
        target_state = self.alg.policy.state_dict()
        target_actor_keys = {key for key in target_state if key.startswith("actor.")}
        source_actor_state = {key: value for key, value in source_state.items() if key.startswith("actor.")}
        if set(source_actor_state) != target_actor_keys:
            missing = sorted(target_actor_keys - set(source_actor_state))
            unexpected = sorted(set(source_actor_state) - target_actor_keys)
            raise RuntimeError(
                f"Parent actor contract mismatch: missing={missing}, unexpected={unexpected}."
            )
        shape_mismatches = {
            key: (tuple(source_actor_state[key].shape), tuple(target_state[key].shape))
            for key in sorted(target_actor_keys)
            if source_actor_state[key].shape != target_state[key].shape
        }
        if shape_mismatches:
            raise RuntimeError(f"Parent actor tensor-shape mismatch: {shape_mismatches}.")

        load_result = self.alg.policy.load_state_dict(source_actor_state, strict=False)
        expected_missing = sorted(set(target_state) - target_actor_keys)
        if load_result.unexpected_keys or sorted(load_result.missing_keys) != expected_missing:
            raise RuntimeError(
                "Actor-only load produced an unexpected state-dict result: "
                f"missing={load_result.missing_keys}, unexpected={load_result.unexpected_keys}."
            )

        if getattr(self.alg, "reference_policy_kl_coef", 0.0) > 0.0:
            self.alg.capture_reference_policy()
        self._warm_start_provenance = {
            "schema_version": 1,
            "path": path,
            "sha256": actual_sha256,
            "parent_iteration": loaded_dict.get("iter"),
            "loaded_components": ["actor"],
            "reinitialized_components": [
                "critic",
                "optimizer",
                "exploration_distribution",
                "curriculum_state",
                "iteration_counter",
            ],
        }
        return dict(self._warm_start_provenance)
