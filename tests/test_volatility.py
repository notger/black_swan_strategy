import unittest
import numpy as np
from volatility import sigma_yearly_from_daily_prices


class TestVolatility(unittest.TestCase):

    def test_output_shape(self):
        test_shape = (10, 3)
        self.assertEqual(
            test_shape,
            sigma_yearly_from_daily_prices(np.random.random(test_shape)).shape
        )

    def test_discontinuous_input(self):
        prices = np.asarray([[1, 1, 1, 1, np.nan, 1, 1], [1, 2, np.nan, np.nan, 3, 4, 5]]).T
        sigma = sigma_yearly_from_daily_prices(prices)

        # Assert that all elements of the first stock are zero, except the nan-element:
        self.assertAlmostEqual(0.0, np.mean(sigma[np.where(~np.isnan(sigma[:, 0])), 0]))

        # Assert that the elements of the second stock are on average above zero:
        self.assertLess(0.0, np.mean(sigma[np.where(~np.isnan(sigma[:, 1])), 1]))

        # Assert that all elements that are nan in the original matrix are nan as well here:
        self.assertEqual(
            len(prices[np.where(np.isnan(prices))]),
            len(sigma[np.where(np.isnan(sigma))])
        )


if __name__ == "__main__":
    unittest.main()
