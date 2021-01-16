#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Methods to calculate yearly volatility based on daily prices
#
# Author: Notger Heinz <notger.heinz@gmail.com>
#

import numpy as np


def sigma_yearly_from_daily_prices(prices):
    """
    Calculate the running yearly volatility from an asset-price-matrix.

    :param prices: np.ndarray of prices. Prices should contain one row per stock, and one column per day.
                   All cells have to be filled.
    :return: np.ndarray, same shape as prices, but filled with the sigma-values.
    """

    assert len(prices.shape) > 1, "prices is not in Matrix shape!"

    N_days = prices.shape[1]

    # Calculate the logarithmic daily return:
    R = np.ones(prices.shape)
    R[:, 1:] = prices[:, 1:] / prices[:, 0:-1]
    Rln = np.log(R)

    # From this, get the standard deviation up to that point:
    sigma = np.zeros(prices.shape)
    for k in range(1, N_days):
        sigma[:, k] = np.std(Rln[:, 0:k + 1], axis=1)

    # And now return the yearly standard deviation from that:
    return sigma * np.sqrt(252)


def generate_stock_prices(num_of_stocks=10, num_of_days=1000):
    """
    Randomly(!) generate a batch of stock prices over a specified horizon.
    This function is useful for general sanity checking and process testing.

    :param num_of_stocks: How many of them?
    :param num_of_days: For how long?
    :return: np.ndarray of shape (num_of_stocks, num_of_days).
    """
    changes = np.random.randn(num_of_stocks * num_of_days).reshape((num_of_stocks, num_of_days))
    prices = changes.cumsum(axis=1) + np.random.random_integers(50, 150, num_of_stocks).reshape((num_of_stocks, 1))

    return np.clip(prices, 1e-6, np.inf)


if __name__ == '__main__':
    """
    The following is a mixture of test and proof of concept / human sanity check.
    As there is no automated test and a human has to have a look at this to say whether this is plausible,
    we do put it here in the main routine and not into a test sub-folder.
    """

    prices = generate_stock_prices()

    s = sigma_yearly_from_daily_prices(prices)

    import matplotlib.pyplot as plt
    plt.figure()
    plt.subplot(211)
    plt.plot(prices.T)
    plt.ylabel("Artifical stock price")
    plt.grid(True)
    plt.subplot(212)
    plt.plot(s.T)
    plt.ylabel("Running yearly volatility")
    plt.xlabel("Time step / day")
    plt.grid(True)
    plt.show()
