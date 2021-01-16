import unittest
from black_scholes_model import get_black_scholes_price


class TestBlackScholesImplementation(unittest.TestCase):

    def test_positive_long_price(self):
        self.assertGreater(get_black_scholes_price(bet_long=True, s0=20, k=100), 0)

    def test_asset_price_increasing_option_price(self):
        self.assertGreater(
            get_black_scholes_price(bet_long=True, s0=200, k=100),
            get_black_scholes_price(bet_long=True, s0=20, k=100)
        )

    # TODO: Add some more tests!


if __name__ == '__main__':
    unittest.main()
