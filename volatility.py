#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Methods to calculate yearly volatility based on daily prices
#
# Author: Notger Heinz <notger@dojomadness.com>
#

import numpy as np

# Prices should contain one row per stock, maximum, and has to be in matrix-shape.
def sigma_yearly_from_daily_prices(prices):
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
    changes = np.random.randn(num_of_stocks * num_of_days).reshape((num_of_stocks, num_of_days))
    prices = changes.cumsum(axis=1) + np.random.random_integers(50, 150, num_of_stocks).reshape((num_of_stocks, 1))

    return np.clip(prices, 1e-6, np.inf)


if __name__ == '__main__':
    prices = generate_stock_prices()

    s = sigma_yearly_from_daily_prices(prices)

    import matplotlib.pyplot as plt
    plt.figure()
    plt.subplot(211)
    plt.plot(prices.T)
    plt.grid(True)
    plt.subplot(212)
    plt.plot(s.T)
    plt.grid(True)
    plt.show()
