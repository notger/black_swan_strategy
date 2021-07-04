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
import typing

import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
import volatility as vol
import black_scholes_model as bsm

MIN_OPTION_PRICE = 1e-3


class SimulationOptions(object):
    def __init__(
            self,
            horizon: int = int(252 / 4),
            minimum_maturity: int = int(50),
            out_of_money_factor: float = 0.7,
            r: float = 0.02,
            bet_long: bool = False,
            min_num_entries: int = 1000,
            remove_discontinuous: bool = False,
            discontinuity_threshold: int = 1,
            path_to_price_files: str = os.environ.get('BLACK_SWAN_PATH', "") + "prices",
    ):
        """
        Please also see the function load_stock_prices for an explanation.

        :param horizon: Investment horizon, i.e. how long will the option we buy be out (=option horizon).
        :param minimum_maturity: Minimum days that a stock has to be alive, before we start trading on it.
        :param out_of_money_factor: How far out of money do we choose our options?
                                    The lower the number below 1.0, the stronger the stock has to move before we
                                    get in the money.
                                    E.g.: val = 0.5 means that for a call option of a stock which currently trades
                                    at 100, we choose a strike price of 200. For a put-option, it would be 5.
        :param r: Risk-free interest rate, used in the Black-Scholes-formula.
        :param bet_long: Are we going long or short, i.e. calls or puts? Has to be boolean.
        :param min_num_entries: Minimum number of entries for a given stock price history.
                                Only relevant during loading of the stock prices!
        :param remove_discontinuous: Should we remove stocks which have missing values in between?
        :param discontinuity_threshold: If those many occur, we remove.
        :param path_to_price_files: Path to the price files.
        """
        self.horizon = int(horizon)
        self.minimum_maturity = minimum_maturity
        self.out_of_money_factor = out_of_money_factor
        self.r = r
        self.bet_long = bet_long
        self.min_num_entries = int(min_num_entries)
        self.remove_discontinuous = remove_discontinuous
        self.discontinuity_threshold = discontinuity_threshold
        self.path_to_price_files = path_to_price_files

    def __str__(self):
        return str(self.__dict__)


class Simulation(object):
    def __init__(self, options: SimulationOptions = SimulationOptions()):
        self.options = options
        self.prices = self.load_stock_prices(
            options.path_to_price_files,
            min_num_entries=options.min_num_entries,
            remove_discontinuous=options.remove_discontinuous,
            discontinuity_threshold=options.discontinuity_threshold
        )
        self.payouts = None
        self.sigmas = None

    @staticmethod
    def is_discontinuous(values, threshold=1):
        """Checks whether a series of values in a np.array is having a NaN in between non-NaN values."""
        nan_entries = np.isnan(values)

        first_proper_value = np.argwhere(~nan_entries).min()
        last_proper_value = np.argwhere(~nan_entries).max()

        # Find all NaN-values:
        nan_args = np.argwhere(nan_entries)

        # If any of the NaN-values are lying after the first proper and before the last proper value,
        # then the data is discontinuous:
        lying_in_between = (nan_args > first_proper_value) * (nan_args < last_proper_value)

        return lying_in_between.sum() >= threshold

    @staticmethod
    def load_stock_prices(
            path="",
            min_num_entries=5000,
            verbose = True,
            remove_discontinuous=False,
            discontinuity_threshold=1
    ):
        """
        Loads all CSVs in a path and extracts the closing-day prices including the dates.
        From this, it constructs a DataFrame with one column per stock and the date as index.

        As we require all dates to be filled, we will prune the dataset so that only continuous dates
        remain.
        # TODO: We have to check whether the dates actually are continuous!

        :param path: Path in which the csv-files will be found.
        :param min_num_entries: All stocks with less entries than these will be scrapped.
        :param remove_discontinuous: Should we remove stocks which have missing values in between?
        :param discontinuity_threshold: If those many occur, we remove.
        :return: pd.DataFrame
        """

        status = ""
        num_rejected = 0

        # No error handling for the following load, as there is no way to recover if that fails.
        for dir_path, dir_names, file_names in os.walk(path):
            for k, f in enumerate(file_names):
                raw = pd.read_csv(os.path.join(dir_path, f))[['Date', 'Close']].set_index('Date').rename(mapper={'Close': f[:-4]}, axis=1)

                # Store data, if the file contains enough entries for us:
                if len(raw) >= min_num_entries:
                    df = raw if k == 0 else df.merge(raw, on='Date', how='outer')
                else:
                    num_rejected += 1

        status += "Loaded {} stocks from {}.\n".format(len(file_names), dir_path)
        status += "{} stocks were rejected for reason of having less than {} entries.\n".format(num_rejected, min_num_entries)

        df = df.sort_index()

        # We do have some misaligned date formats, where the date is given in the form of DD-"month"-YY.
        # Fixing this here is too cumbersome, so we will throw those away.
        def year_from_date(df_row):
            # Extract the year from the row's Index value and return the year or -1, so that we can later filter.
            try:
                return int(df_row.Index[:4])
            except:
                return -1

        df['year'] = [year_from_date(row) for row in df.itertuples()]

        status += "After filtering out {} rows with weird date formats, {} rows remain.\n".format(
            len(df.query('year <= 0')),
            len(df.query('year > 0')),
        )

        df = df.query('year > 0').drop('year', axis=1)

        # Now we have to filter for non-continuous time-lines:
        if remove_discontinuous:
            df = df[
                [col for col in df if not Simulation.is_discontinuous(df[col].values, threshold=discontinuity_threshold)]
            ]

        status += "After removal of discontinuous stock price lists{}, {} stocks are left.\n".format(
            '' if remove_discontinuous else ' was skipped',
            len(df.columns)
        )

        if verbose:
            print(status)

        return df

    @staticmethod
    def _calculate_sigmas_wrapper(prices):
        """
        Wrapper for easier debugging.
        :param prices: Pandas dataframe of the prices that you want sigmas for. Make sure it is a dataframe, not a series!
        :return: pd.DataFrame
        """
        return pd.DataFrame(
            vol.sigma_yearly_from_daily_prices(prices.values),
            index=prices.index,
            columns=prices.columns
        )

    @staticmethod
    def get_last_index_within_range(array, max_index):
        """
        Helper function to find the last valid entry in an array.

        We do this by first checking for all non-nan-entries within the range up to and including max_index,
        then getting the indices of the occurences of all the non-nan-entries and choosing the max-value.
        """
        return np.max(np.where(~np.isnan(array[:max_index + 1])), initial=0)

    @staticmethod
    def get_invested_value(
            prices=None,
            sigmas=None,
            index=0,
            horizon=252,
            out_of_money_factor=0.8,
            r=0.02,
            bet_long=False,
    ):
        """
        :param prices: Array of historical prices for that commodity. Prices must only have one dimension!
        :param sigmas: Pre-calculated yearly sigmas.
        :param index: Number of day that we make the investment decision.
        :param horizon: Investment horizon in days (252 = 1 year).
        :param out_of_money_factor: Which factor of the current price should the strike price be?
                                    A value of 1 indicates strike-price = current price,
                                    a value of 2 indicates a strike-price of zero for call-options
                                    and double the current price for put-options.
        :param r: Risk-free return rate.
        :param bet_long: Should we go long?
        :return: Payout at the end of the option.
        """

        try:
            assert len(prices.shape) == 1, "Prices has to be of shape (-1,), but is of shape {}".format(prices.shape)
            assert len(sigmas.shape) == 1, "Sigmas has to be of shape (-1,), but is of shape {}".format(sigmas.shape)
        except AssertionError as e:
            raise e

        if prices[index] == np.nan:
            raise ValueError(f'Calculation requested for non-existing stock-price = np.nan at index {index}.')

        # Calculate the strike price at which we want to buy the option.
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

        # As we are going to divide by the option price later to get the ROI, it better be noticeably
        # above zero. Also, an option price very close to zero is indicative of a stock price very close
        # to zero. To be safe and operate in sane regions, we will not trade on this:
        if option_price < MIN_OPTION_PRICE:
            #raise ValueError(f"Option price {option_price} is below threshold {MIN_OPTION_PRICE}.")
            return np.nan

        # Get final payout, depending on whether the option ran out and was realised, or not.
        # Get the payout index from getting the last index that has a value in our pricing array.
        payout_index = Simulation.get_last_index_within_range(prices, index + horizon)
        realised = (index + horizon) <= len(prices) - 1

        # If the option realised, i.e. if it ran out, we get the final value paid out.
        # As we are interested in the relative return on investment, that is what we are returning here.
        if realised:
            if bet_long:
                return (max(0, prices[payout_index] - strike_price) - option_price) / option_price
            else:
                return (max(0, strike_price - prices[payout_index]) - option_price) / option_price
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

    @staticmethod
    def _run(prices, options, verbose=True, get_invested_value=None):
        """
        Runs the simulation by taking the stock closing prices, calculating the volatilities and going over all
        possible investment starting points to calculate the end-of-investment return.

        The return value is a DataFrame with dates as indices and the return on investment for each
        starting date and for each stock, where stocks are ordered column-wise.

        :return: pd.DataFrame
        """
        # For each stock, calculate the running yearly volatility:
        sigmas = Simulation._calculate_sigmas_wrapper(prices)

        if verbose:
            print("Finished calculating the yearly sigmas.")

        # Then extract the values and call the ROI-calculation for investing one given day:
        payouts = np.empty_like(prices.values)
        payouts[:] = np.nan

        for stock_idx, stock in tqdm(enumerate(prices.columns), total=len(prices.columns)):
            # We do not want to start trading right away, when a stock becomes available. So we need to find out
            # the "date of birth" for a stock (which is the minimum index where an isna-comparison changes from
            # True to False) and then add our minimum maturity.
            # Alternative for date_of_stock_trading_start: np.argmin(prices[stock].isna().values), which is a bit less
            # readable.
            #date_of_stock_trading_start = prices[stock].isna().idxmin()
            date_of_stock_trading_start = np.argmin(prices[stock].isna().values)
            #trading_start_idx = prices.index.get_loc(date_of_stock_trading_start) + options.minimum_maturity
            trading_start_idx = date_of_stock_trading_start + options.minimum_maturity

            for k in range(trading_start_idx, len(sigmas)):
                try:
                    payouts[k, stock_idx] = get_invested_value(
                        prices=prices[stock].values.squeeze(),
                        sigmas=sigmas[stock].values.squeeze(),
                        index=k,
                        horizon=options.horizon,
                        out_of_money_factor=options.out_of_money_factor,
                        r=options.r,
                        bet_long=options.bet_long,
                    )
                except AssertionError as a:
                    # Raise and enhanced assertion error with info that was not available within the
                    # get_invested_value-function:
                    s = f'Encountered AssertionError for improper shape of vectors ' \
                        f'prices/sigmas for {stock} (idx: {stock_idx}) in step {k}: {a}\n'
                    s += f'Shapes passed were: {prices[stock].values.squeeze().shape} for prices ' \
                         f'and {sigmas[stock].values.squeeze().shape} for sigmas.\n'
                    raise AssertionError(s)
                except ValueError as v:
                    # Raise an enhanced value error with info that was not available within the get_invested_value-function:
                    s = f'Encountered ValueError "{v}" for {stock} (idx: {stock_idx}) in step {k}.\n'
                    s += f'Stock price at index {k} was {prices[stock].values.squeeze()[k]}.\n'
                    s += f'Sigma at index {k} was {sigmas[stock].values.squeeze()[k]}.\n'
                    s += f'Passed parameters were:\n{options}\n'
                    raise ValueError(s)
                except Exception as e:
                    raise Exception("Unknown exception raised.")

        pay_outs = pd.DataFrame(payouts, index=sigmas.index, columns=sigmas.columns)

        # Store the results in a separate dataframe, where entries are nan where we could not calculate the ROI.
        return pay_outs, sigmas

    def run(self, verbose=True, defined_subset: typing.List[str] = [], random_subset_size: int = 0):
        # If the set has been defined, take the defined set (if it exists in the column set),
        # otherwise check if random_subset_size has been chosen to be larger than 0, generate a subset:
        if len(defined_subset) > 0:
            selected_stocks = defined_subset
            assert set(defined_subset).issubset(set(self.prices.columns)), \
                f'Defined set of columns {defined_subset} not in set of price-columns {self.prices.columns}.'
        elif random_subset_size > 0:
            import random
            selected_stocks = random.sample(list(self.prices.columns), k=random_subset_size)
        else:
            selected_stocks = self.prices.columns

        self.payouts, self.sigmas = self._run(
            self.prices[selected_stocks],
            self.options,
            verbose=verbose,
            get_invested_value=Simulation.get_invested_value
        )
        return self.payouts
