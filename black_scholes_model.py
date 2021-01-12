#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Methods to calculate option pricing as per Black-Scholes-Formula
#
# Author: Notger Heinz <notger@dojomadness.com>
#

import numpy as np
import scipy.stats as ss


# Black and Scholes from https://en.wikipedia.org/wiki/Black%E2%80%93Scholes_model#Black.E2.80.93Scholes_equation:
def d1(s0, k, r, sigma, t):
    return (np.log(s0 / k) + (r + (sigma ** 2) / 2) * t) / (sigma * np.sqrt(t))


def d2(s0, k, r, sigma, t):
    # The following is equivalent to (np.log(s0 / k) + (r - (sigma ** 2) / 2) * t) / (sigma * np.sqrt(t)),
    # but more readable and directly from the wiki:
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
    assert get_black_scholes_price(bet_long=True, s0=20, k=100) > 0
    assert get_black_scholes_price(bet_long=True, s0=200, k=100) > get_black_scholes_price(bet_long=True, s0=20, k=100)

    # Some more specialised tests:
    # For a stock price for Deutsche Bank of 15.395, a Vola250 of 33.65%, SD250 = 1.14, we will check whether we can
    # replicate the following real option prices:
    current_price = 15.395
    sigma = 0.3365
    # Type - strike price - time to fulfillment in trading days - option price to achieve
    # Call on 15.4, 13d -> 0.25
    print("{} == {} ?".format(
        0.25, get_black_scholes_price(bet_long=True, s0=current_price, sigma=sigma, t=13/252, k=15.4)
    ))
    # Put on 15.4, 13d -> 0.17
    print("{} == {} ?".format(
        0.17, get_black_scholes_price(bet_long=False, s0=current_price, sigma=sigma, t=13/252, k=15.4)
    ))
    # Call on 14.5, 13d -> 0.93
    # Put on 14.0, 13d -> 0.01
    # Put on 15.17, 33d -> 0.35
    # Call on 15.50, 33d -> 0.50
    # Put on 17.0, 33d -> 1.62

    # Some random stuff from the internet (http://www.option-price.com/index.php):
    # (Or here: http://www.fintools.com/resources/online-calculators/options-calcs/options-calculator/.)
    print("{} == {} ?".format(
        3.063, get_black_scholes_price(bet_long=True, s0=100, sigma=0.25, t=30/252, k=100)
    ))
    print("{} == {} ?".format(
        2.652, get_black_scholes_price(bet_long=False, s0=100, sigma=0.25, t=30/252, k=100)
    ))
    print("{} == {} ?".format(
        0.227, get_black_scholes_price(bet_long=True, s0=50, sigma=0.30, t=60/252, k=60)
    ))
    print("{} == {} ?".format(
        9.736, get_black_scholes_price(bet_long=False, s0=50, sigma=0.30, t=60/252, k=60)
    ))
