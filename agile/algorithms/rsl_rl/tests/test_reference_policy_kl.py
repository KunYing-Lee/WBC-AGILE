"""Unit tests for fixed reference-policy regularization."""

from __future__ import annotations

import unittest

import torch
from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCritic


class TestReferencePolicyKl(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.policy = ActorCritic(
            6,
            8,
            3,
            init_noise_std=0.35,
            noise_std_type="log",
            log_std_range=(-3.0, -0.5),
            actor_hidden_dims=[16, 8],
            critic_hidden_dims=[16, 8],
            activation="elu",
        )

    def test_reference_policy_is_frozen_and_detects_actor_drift(self) -> None:
        algorithm = PPO(self.policy, schedule="fixed", reference_policy_kl_coef=1.0)
        algorithm.capture_reference_policy()
        observations = torch.randn(32, 6)

        algorithm.policy.update_distribution(observations)
        algorithm.reference_policy.update_distribution(observations)
        initial_kl = torch.distributions.kl_divergence(
            algorithm.policy.distribution,
            algorithm.reference_policy.distribution,
        ).sum(dim=-1)
        torch.testing.assert_close(initial_kl, torch.zeros_like(initial_kl), atol=1.0e-6, rtol=0.0)
        self.assertFalse(any(parameter.requires_grad for parameter in algorithm.reference_policy.parameters()))

        with torch.no_grad():
            algorithm.policy.actor.layers[-1].bias[0] += 0.1
        algorithm.policy.update_distribution(observations)
        algorithm.reference_policy.update_distribution(observations)
        drift_kl = torch.distributions.kl_divergence(
            algorithm.policy.distribution,
            algorithm.reference_policy.distribution,
        ).sum(dim=-1)
        self.assertGreater(drift_kl.mean().item(), 0.0)

    def test_reference_policy_parameters_are_validated(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-negative"):
            PPO(self.policy, schedule="fixed", reference_policy_kl_coef=-0.1)
        for fraction in (0.0, 1.1):
            with self.subTest(fraction=fraction), self.assertRaisesRegex(ValueError, "CVaR fraction"):
                PPO(
                    self.policy,
                    schedule="fixed",
                    reference_policy_kl_coef=1.0,
                    reference_policy_kl_cvar_fraction=fraction,
                )

    def test_critic_warmup_transition_learning_rate_is_configurable_and_validated(self) -> None:
        algorithm = PPO(
            self.policy,
            schedule="fixed",
            critic_warmup_steps=20,
            critic_warmup_transition_learning_rate=1.0e-6,
        )
        self.assertEqual(algorithm.critic_warmup_steps, 20)
        self.assertEqual(algorithm.critic_warmup_transition_learning_rate, 1.0e-6)

        with self.assertRaisesRegex(ValueError, "must be positive"):
            PPO(
                self.policy,
                schedule="fixed",
                critic_warmup_steps=20,
                critic_warmup_transition_learning_rate=0.0,
            )


if __name__ == "__main__":
    unittest.main()
