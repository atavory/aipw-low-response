from __future__ import annotations

import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from weighted_residual_selection import select_gamma  # noqa: E402


class WeightedResidualSelectionTest(unittest.TestCase):
    def test_selects_clear_improvement(self) -> None:
        result = select_gamma(
            outcomes=[1.0, 2.0, 3.0, 4.0],
            responses=[1, 1, 1, 1],
            response_predictions=[0.5] * 4,
            candidate_predictions={
                0.0: [0.0, 0.0, 0.0, 0.0],
                1.0: [1.0, 2.0, 3.0, 4.0],
            },
            gamma_grid=(0.0, 1.0),
        )
        self.assertEqual(1.0, result.gamma)

    def test_stands_down_without_clear_improvement(self) -> None:
        result = select_gamma(
            outcomes=[-1.0, 1.0, -1.0, 1.0],
            responses=[1, 1, 1, 1],
            response_predictions=[0.5] * 4,
            candidate_predictions={
                0.0: [0.0, 0.0, 0.0, 0.0],
                1.0: [-0.1, -0.1, 0.1, 0.1],
            },
            gamma_grid=(0.0, 1.0),
        )
        self.assertEqual(0.0, result.gamma)

    def test_rejects_invalid_response_predictions(self) -> None:
        with self.assertRaisesRegex(ValueError, "response predictions"):
            select_gamma(
                outcomes=[1.0, 2.0],
                responses=[1, 1],
                response_predictions=[0.0, 0.5],
                candidate_predictions={0.0: [0.0, 0.0], 1.0: [1.0, 1.0]},
                gamma_grid=(0.0, 1.0),
            )


if __name__ == "__main__":
    unittest.main()

