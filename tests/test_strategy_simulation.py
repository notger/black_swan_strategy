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

    def test_get_last_index_within_range(self):
        # Create a test array which contains two nan-values at index six and seven (zero-based, ofc ;) ).
        array = np.asarray([1, 1, 1, 1, 1, 1, np.nan, np.nan, 1, 1])

        self.assertEqual(4, Simulation.get_last_index_within_range(array, 4))
        self.assertEqual(5, Simulation.get_last_index_within_range(array, 6))
        self.assertEqual(5, Simulation.get_last_index_within_range(array, 7))
        self.assertEqual(8, Simulation.get_last_index_within_range(array, 8))
        self.assertEqual(9, Simulation.get_last_index_within_range(array, 100))
        self.assertEqual(np.int64, type(Simulation.get_last_index_within_range(array, 100)))

        nan_array = np.empty_like(array)
        nan_array[:] = np.nan
        self.assertEqual(0, Simulation.get_last_index_within_range(nan_array, 100))


if __name__ == '__main__':
    unittest.main()
