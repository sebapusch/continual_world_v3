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

    def test_ordered_test_envs_share_training_one_hot_encodings(self) -> None:
        env = get_cl_env(
            ["push-v3", "window-close-v3"],
            steps_per_task=1,
            episode_horizon=10,
            seed=7,
        )
        try:
            self.assertEqual(len(env.test_envs), 2)
            self.assertIs(env.current_test_env, env.test_envs[0])
            self.assertIs(env.test_env, env.test_envs[0])

            train_observation, _ = env.reset(seed=7)
            first_test_observation, _ = env.test_envs[0].reset(seed=7)
            second_test_observation, _ = env.test_envs[1].reset(seed=7)
            np.testing.assert_array_equal(
                first_test_observation[-2:], train_observation[-2:]
            )
            np.testing.assert_array_equal(second_test_observation[-2:], [0.0, 1.0])

            env.step(env.action_space.sample())
            self.assertIs(env.current_test_env, env.test_envs[1])
            self.assertIs(env.test_env, env.test_envs[1])
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
