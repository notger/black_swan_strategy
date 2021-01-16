#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Simulates a black swan strategy:
#   For every stock, at every trading day, we invest a certain amount into push-options.
#   We are choosing push-options that are out of money, where the out of money-ratio is
#   a parameter to our strategy. The other parameter is the realisation period that we want
#   to have.
#   The value of this invested amount gets paid out once the option runs out
#   (or we arrive in the present).
#
# Author: Notger Heinz <notger.heinz@gmail.com>
#

import os
import numpy as np
import pandas as pd
import volatility as vol
import black_scholes_model as bsm


def get_invested_value(prices=None, sigmas=None, index=0, horizon=252, out_of_money_factor=0.8, r=0.05, bet_long=False):
    """
    :param prices: Array of historical prices for that commodity. Prices must only have one dimension!
    :param sigmas: Pre-calculated yearly sigmas.
    :param index: Number of day that we make the investment decision.
    :param horizon: Investment horizon in days (252 = 1 year).
    :param out_of_money_factor: Which factor of the current price should the strike price be?
    :param r: Risk-free return rate.
    :param bet_long: Should we go long?
    :return: Payout at the end of the option.
    """

    assert len(prices.shape) == 1, "Prices has to be of shape (-1,), but is of shape {}".format(prices.shape)
    assert len(sigmas.shape) == 1, "Prices has to be of shape (-1,), but is of shape {}".format(sigmas.shape)

    strike_price = (2 - out_of_money_factor) * prices[index] if bet_long else out_of_money_factor * prices[index]

    # Get current option price at the given index:
    option_price = bsm.get_black_scholes_price(
        bet_long=bet_long,
        s0=prices[index],
        k=strike_price,
        r=r,
        sigma=sigmas[index],
        t=horizon/252
    )

    # Get final payout, depending on whether the option ran out and was realised, or not:
    payout_index = min(index + horizon, len(prices) - 1)
    realised = (index + horizon) <= len(prices) - 1

    # DEBUG:
    # TODO: Do we need this, should this be done someplace else and in which way?
    if 1500 < index < 2000:
        s = "k = {}:".format(index)
        s += "Current price is {} -> option price = {}, pay-out-index-price will be {}, strike-price is {}.".format(
            prices[index], option_price, prices[payout_index], out_of_money_factor * prices[index]
        )
        s += "\nDifference to strike price at end of period is: {}".format(
            max(0, out_of_money_factor * prices[index] - prices[payout_index]))
        s += " Yearly sigma is: {}".format(sigmas[index])
        print(s)
        print()

    # If the option realised, i.e. if it ran out, we get the final value paid out.
    # As we are interested in the relative return on investment, that is what we are returning here.
    if realised:
        if bet_long:
            return (max(0, prices[payout_index] - out_of_money_factor * prices[index]) - option_price) / option_price
        else:
            return (max(0, out_of_money_factor * prices[index] - prices[payout_index]) - option_price) / option_price
    else:
        remaining_horizon = index + horizon - len(prices) + 1
        option_price_end = bsm.get_black_scholes_price(
            bet_long=bet_long,
            s0=prices[-1],
            k=out_of_money_factor * prices[index],
            r=r,
            sigma=sigmas[-1],
            t=remaining_horizon/252
        )
        return (option_price_end - option_price) / option_price


def load_stock_prices(path=""):
    # TODO: Needs documentation, error handling and a more flexible import routine.
    # Currently assumes that you get a download of the NYSE-prices in a certain format.
    raw = pd.read_csv(path)

    stocks = {}
    for col in raw.columns:
        if col == "PriceNumber" or col == "Date":
            continue
        else:
            stocks[col] = raw[col].values.copy()

    del raw
    return stocks


if __name__ == '__main__':
    # TODO: Clean up, compartimentalise, outsource and give the user a choice via input parameter.

    horizon = int(252/4)
    out_of_money_factor = 0.7
    r = 0.05
    bet_long = False

    # Load all stock information that we have:
    # prices = load_stock_prices(random=False)
    prices = load_stock_prices(path=os.path.expanduser("prices/dtegy.csv"))

    # DEBUG:
    stock_list_to_keep = ['hp']#'dow', 'tex', 'hp']
    prices = {x: prices[x] for x in stock_list_to_keep if x in prices}

    # Get the sigmas:
    sigmas = {}
    for stock in prices.keys():
        sigmas[stock] = vol.sigma_yearly_from_daily_prices(prices[stock].reshape((1, -1))).reshape((-1,))

    pay_out = {}

    for s, stock in enumerate(prices.keys()):
        num_stock_prices = len(prices[stock])
        pay_out[stock] = np.zeros(num_stock_prices)
        for k in range(100, num_stock_prices):
            pay_out[stock][k] = get_invested_value(
                prices=prices[stock], sigmas=sigmas[stock], index=k,
                horizon=int(252/4), out_of_money_factor=out_of_money_factor, bet_long=bet_long
            )

    acc_pay_out = {}
    for stock in pay_out.keys():
        acc_pay_out[stock] = np.cumsum(pay_out[stock])

    final_pay_out = [p[-2] for s, p in acc_pay_out.items()]
    mean_final_pay_out = np.mean(final_pay_out)

    print("Final pay_outs: {}".format(final_pay_out))

    import matplotlib.pyplot as plt
    plt.subplot(311)
    for stock in pay_out.keys():
        plt.plot(np.cumsum(pay_out[stock]), label=stock)
    plt.title('Mean final payout: {:.2f} % for horizon of {} days, r = {:.1f} %, ofmf = {:.2f} and {} strategy'.format(
        100 * mean_final_pay_out, horizon, 100 * r, out_of_money_factor, "long" if bet_long else "short"))
    plt.ylabel('accumulated fonds value')
    plt.legend(loc=2)
    plt.grid(True)
    plt.subplot(312)
    for stock in prices.keys():
        plt.plot(prices[stock], label=stock)
    plt.ylabel('stock price')
    plt.legend(loc=2)
    plt.grid(True)
    plt.subplot(313)
    x = np.linspace(0, len(pay_out) - 1, len(pay_out))
    plt.plot(x, final_pay_out)
    plt.xticks(x, pay_out.keys())
    plt.ylabel('final payout per stock')
    plt.xlabel('stock')
    plt.grid(True)
    plt.show()
