# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the RSL-RL command-line configuration overrides."""

import argparse
import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace


def _load_cli_args_module():
    module_path = Path(__file__).parents[1] / "scripts" / "cli_args.py"
    spec = importlib.util.spec_from_file_location("wbc_agile_cli_args", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cli_args = _load_cli_args_module()


def _args(device):
    return argparse.Namespace(
        seed=None,
        device=device,
        resume=None,
        load_run=None,
        checkpoint=None,
        run_name=None,
        logger=None,
        log_project_name=None,
    )


class TestUpdateRslRlCfg(unittest.TestCase):
    def test_device_override_keeps_sim_and_agent_on_the_same_gpu(self):
        cfg = SimpleNamespace(device="cuda:0", logger="tensorboard")

        updated = cli_args.update_rsl_rl_cfg(cfg, _args("cuda:7"))

        self.assertEqual(updated.device, "cuda:7")

    def test_missing_device_override_preserves_configured_device(self):
        cfg = SimpleNamespace(device="cuda:3", logger="tensorboard")

        updated = cli_args.update_rsl_rl_cfg(cfg, _args(None))

        self.assertEqual(updated.device, "cuda:3")


if __name__ == "__main__":
    unittest.main()
