"""Numerical-safety tests for the SAC learner."""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from rl.algorithms.sac import SACLearner
from rl.datasets.dataset import Batch


class SACNumericalSafetyTest(unittest.TestCase):
    def _learner(self) -> SACLearner:
        return SACLearner(
            seed=0,
            observations=jnp.zeros((1, 3)),
            actions=jnp.zeros((1, 2)),
            hidden_dims=(8, 8),
        )

    def test_nonfinite_action_uses_bounded_fallback(self) -> None:
        learner = self._learner()
        nan_params = jax.tree_util.tree_map(
            lambda value: jnp.full_like(value, jnp.nan), learner.actor.params
        )
        learner.actor = learner.actor.replace(params=nan_params)

        with self.assertWarnsRegex(RuntimeWarning, "fallback action"):
            action = learner.sample_actions(
                np.zeros(3, dtype=np.float32), deterministic=True
            )

        np.testing.assert_array_equal(action, np.zeros(2))
        self.assertEqual(learner.nonfinite_action_count, 1)

    def test_nonfinite_batch_skips_update(self) -> None:
        learner = self._learner()
        batch = Batch(
            observations=np.full((2, 3), np.nan, dtype=np.float32),
            actions=np.zeros((2, 2), dtype=np.float32),
            rewards=np.zeros(2, dtype=np.float32),
            masks=np.ones(2, dtype=np.float32),
            next_observations=np.zeros((2, 3), dtype=np.float32),
        )

        with self.assertWarnsRegex(RuntimeWarning, "Skipping SAC update"):
            info = learner.update(batch)

        self.assertEqual(info["update_skipped"], 1.0)
        self.assertEqual(learner.skipped_updates, 1)


if __name__ == "__main__":
    unittest.main()
