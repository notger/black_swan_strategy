import unittest
import numpy as np
from strategy_simulation import is_discontinuous


class TestStrategySimulation(unittest.TestCase):

    def test_is_discontinuous(self):
        self.assertFalse(is_discontinuous(np.asarray([1, 1, np.nan, np.nan])))
        self.assertFalse(is_discontinuous(np.asarray([np.nan, np.nan, 1, 1, ])))
        self.assertFalse(is_discontinuous(np.asarray([np.nan, np.nan, 1, 1, np.nan, np.nan])))
        self.assertFalse(is_discontinuous(np.asarray([1])))
        self.assertFalse(is_discontinuous(np.asarray([1, np.nan])))
        self.assertFalse(is_discontinuous(np.asarray([np.nan, 1, np.nan])))

        self.assertTrue(is_discontinuous(np.asarray([np.nan, np.nan, 1, 1, np.nan, 1])))
        self.assertTrue(is_discontinuous(np.asarray([1, 1, np.nan, 1, 1])))

        # TODO: Add test for threshold


if __name__ == '__main__':
    unittest.main()
