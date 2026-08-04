"""Small integration checks against the installed Meta-World package."""

import unittest

import numpy as np

from continualworld import CW10, CW20, get_cl_env


class ContinualWorldEnvTest(unittest.TestCase):
    def test_canonical_sequences_use_v3_names(self) -> None:
        self.assertEqual(len(CW10), 10)
        self.assertEqual(CW20, CW10 + CW10)
        self.assertTrue(all(name.endswith("-v3") for name in CW20))

    def test_one_hot_task_switch_and_exhaustion(self) -> None:
        env = get_cl_env(
            ["push-v3", "window-close-v3"],
            steps_per_task=1,
            episode_horizon=10,
            seed=5,
        )
        try:
            observation, info = env.reset(seed=5)
            self.assertTrue(env.observation_space.contains(observation))
            np.testing.assert_array_equal(observation[-2:], [1.0, 0.0])
            self.assertEqual(info["task_name"], "push-v3")

            _, _, _, truncated, info = env.step(env.action_space.sample())
            self.assertTrue(truncated)
            self.assertTrue(info["task_boundary"])
            self.assertEqual(env.current_task_name, "window-close-v3")

            observation, _ = env.reset()
            np.testing.assert_array_equal(observation[-2:], [0.0, 1.0])
            _, _, _, truncated, _ = env.step(env.action_space.sample())
            self.assertTrue(truncated)
            self.assertTrue(env.exhausted)
            self.assertEqual(env.total_steps, 2)
            with self.assertRaisesRegex(RuntimeError, "exhausted"):
                env.reset()
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
