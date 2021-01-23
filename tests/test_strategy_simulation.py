import unittest
import numpy as np
from strategy_simulation import Simulation


class TestStrategySimulation(unittest.TestCase):

    def test_is_discontinuous(self):
        self.assertFalse(Simulation.is_discontinuous(np.asarray([1, 1, np.nan, np.nan])))
        self.assertFalse(Simulation.is_discontinuous(np.asarray([np.nan, np.nan, 1, 1, ])))
        self.assertFalse(Simulation.is_discontinuous(np.asarray([np.nan, np.nan, 1, 1, np.nan, np.nan])))
        self.assertFalse(Simulation.is_discontinuous(np.asarray([1])))
        self.assertFalse(Simulation.is_discontinuous(np.asarray([1, np.nan])))
        self.assertFalse(Simulation.is_discontinuous(np.asarray([np.nan, 1, np.nan])))

        self.assertTrue(Simulation.is_discontinuous(np.asarray([np.nan, np.nan, 1, 1, np.nan, 1])))
        self.assertTrue(Simulation.is_discontinuous(np.asarray([1, 1, np.nan, 1, 1])))

        # Threshold-tests:
        self.assertFalse(Simulation.is_discontinuous(np.asarray([1, 1, np.nan, 1, 1]), threshold=2))
        self.assertFalse(Simulation.is_discontinuous(np.asarray([1, 1, np.nan, 1, 1]), threshold=3))
        self.assertTrue(Simulation.is_discontinuous(np.asarray([1, 1, np.nan, np.nan, np.nan, 1, 1]), threshold=3))

if __name__ == '__main__':
    unittest.main()
