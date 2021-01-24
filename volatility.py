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

    Note that since the asset-price-matrix can contain nan-values, the standard deviations for days where the asset
    was not traded can not be calculated. The sigma-matrix will thus also contain nan-values in those spots.
    However, for an entry after a missing entry, we will pretend that the line of values was continuous and
    thus get a volatility estimation here. It basically is like if the asset never had a nan-value in the first place.

    :param prices: np.ndarray of prices. Prices should contain one column per stock, and one row per day.
    :return: np.ndarray, same shape as prices, but filled with the sigma-values.
    """

    assert len(prices.shape) > 1, "prices is not in Matrix shape!"

    N_days = prices.shape[0]
    N_stocks = prices.shape[1]

    # Calculate the logarithmic daily return:
    R = np.ones(prices.shape)
    R[1:, :] = prices[1:, :] / prices[0:-1, :]
    Rln = np.log(R)

    # TODO: Is this correct? Do we really take the standard deviation from all of history to calculate the yearly standard deviation or rather the standard deviation over the last year?
    # From this, get the standard deviation up to that point.
    # Unfortunately, the price-matrix does not have to be filled for any day, so we have to deal with nan-values.
    # This means we can't just vectorise the whole matrix, but have to go stock by stock:
    sigma = np.zeros(prices.shape)
    for stock in range(N_stocks):
        for day in range(1, N_days):
            # The following is going to be a bit ugly to parse:
            # - Use where to find the non-nan-values in Rln for that given stock.
            # - With these indices, select the non-nan-values in Rln and get the standard deviation on it.
            if np.isnan(prices[day, stock]):
                sigma[day, stock] = np.nan
            else:
                sigma[day, stock] = np.std(
                    Rln[
                        np.where(~np.isnan(Rln[0:day + 1, stock])), stock
                    ]
                )

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
    prices = changes.cumsum(axis=1) + np.random.randint(50, 150, num_of_stocks).reshape((num_of_stocks, 1))

    return np.clip(prices, 1e-6, np.inf).T


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
    plt.plot(prices)
    plt.ylabel("Artifical stock price")
    plt.grid(True)
    plt.subplot(212)
    plt.plot(s)
    plt.ylabel("Running yearly volatility")
    plt.xlabel("Time step / day")
    plt.grid(True)
    plt.show()
