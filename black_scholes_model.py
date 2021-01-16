#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Methods to calculate option pricing as per Black-Scholes-Formula
# Black and Scholes from https://en.wikipedia.org/wiki/Black%E2%80%93Scholes_model#Black.E2.80.93Scholes_equation
#
# Author: Notger Heinz <notger.heinz@gmail.com>
#

import numpy as np
import scipy.stats as ss


def d1(s0, k, r, sigma, t):
    """
    For an explanation of the parameters, please see the get_black_scholes_price-function below.
    :return: float
    """
    return (np.log(s0 / k) + (r + (sigma ** 2) / 2) * t) / (sigma * np.sqrt(t))


def d2(s0, k, r, sigma, t):
    """
    For an explanation of the parameters, please see the get_black_scholes_price-function below.
    The following is equivalent to (np.log(s0 / k) + (r - (sigma ** 2) / 2) * t) / (sigma * np.sqrt(t)),
    but more readable and directly from the wiki:
    :return: float
    """
    return d1(s0, k, r, sigma, t) - sigma * np.sqrt(t)


def get_black_scholes_price(bet_long=True, s0=100.0, k=100.0, r=0.05, sigma=0.3, t=0.15):
    """
    Only works for european type options!

    :param bet_long: Go long or not?
    :param s0: Current stock price.
    :param k: Underlying strike price.
    :param r: Risk-free return.
    :param sigma: Standard deviation of the stock over one year.
    :param t: time until fulfilment, expressed in years.
    :return: float
    """
    if bet_long:  # = if we price a call option:
        return s0 * ss.norm.cdf(d1(s0, k, r, sigma, t)) - k * np.exp(-r * t) * ss.norm.cdf(d2(s0, k, r, sigma, t))
    else:  # = if we price a put option
        return k * np.exp(-r * t) * ss.norm.cdf(-d2(s0, k, r, sigma, t)) - s0 * ss.norm.cdf(-d1(s0, k, r, sigma, t))


if __name__ == '__main__':
    # Some tests that have to be interpreted by a human:
    def print_pricing_test(supposed_option_price=0.0, current_asset_price=0.0, sigma=0.0, bet_long=True, t=0.0, k=15.4):
        print("{} == {} ?".format(
            supposed_option_price, get_black_scholes_price(bet_long=bet_long, s0=current_asset_price, sigma=sigma, t=t, k=k)
        ))
    # For a stock price for Deutsche Bank of 15.395, a Vola250 of 33.65%, SD250 = 1.14, we will check whether we can
    # replicate the following real option prices:
    current_price = 15.395
    sigma = 0.3365
    # Type - strike price - time to fulfillment in trading days - option price to achieve
    # Call on 15.4, 13d -> 0.25
    print_pricing_test(supposed_option_price=0.25, current_asset_price=current_price, sigma=sigma, bet_long=True, t=13/252, k=15.4)
    # Put on 15.4, 13d -> 0.17
    print_pricing_test(supposed_option_price=0.17, current_asset_price=current_price, sigma=sigma, bet_long=False, t=13/252, k=15.4)
    # Other stuff (currently un-tested):
    # Call on 14.5, 13d -> 0.93
    # Put on 14.0, 13d -> 0.01
    # Put on 15.17, 33d -> 0.35
    # Call on 15.50, 33d -> 0.50
    # Put on 17.0, 33d -> 1.62

    # Some random stuff from the internet (http://www.option-price.com/index.php):
    # (Or here: http://www.fintools.com/resources/online-calculators/options-calcs/options-calculator/.)
    print_pricing_test(supposed_option_price=3.063, current_asset_price=100, bet_long=True, sigma=0.25, t=30/252, k=100)
    print_pricing_test(supposed_option_price=2.652, current_asset_price=100, bet_long=False, sigma=0.25, t=30/252, k=100)
    print_pricing_test(supposed_option_price=0.227, current_asset_price=50, bet_long=True, sigma=0.30, t=60 / 252, k=60)
    print_pricing_test(supposed_option_price=9.736, current_asset_price=50, bet_long=False, sigma=0.30, t=60 / 252, k=60)
