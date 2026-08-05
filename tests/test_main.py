"""Tests for the training schedule used by the example entry point."""

import unittest

from main import _should_train


class TrainingScheduleTest(unittest.TestCase):
    def test_training_starts_on_time_and_repeats_at_the_interval(self) -> None:
        due = [
            timestep
            for timestep in range(1, 5_001)
            if _should_train(
                timestep,
                training_starts=1_000,
                gradient_update_interval=1_000,
            )
        ]

        self.assertEqual(due, [1_000, 2_000, 3_000, 4_000, 5_000])


if __name__ == "__main__":
    unittest.main()
