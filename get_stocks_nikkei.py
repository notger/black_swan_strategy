from datetime import date
import pandas as pd
import numpy as np
import yfinance as yf
import os
today = date.today()

def get_nikkei225_stocks():
    #getting all 255 nikkei company and ticker symbols form a dedicated webpage
    #returns a dictionary with the company name without spaces as keys and the ticker usable as yfinance input as values
    stocks = {}
    df = pd.read_html('https://topforeignstocks.com/indices/the-components-of-the-nikkei-225-index/')[0]
    for i in range(len(df)):
        stocks[df['Company Name'][i]] = str(df['Ticker'][i]) + '.T'
    return stocks


def get_stock_prices(stocks, datestart = '1980-01-01', dateend = today.strftime('%Y-%m-%d')):
    #downloading all stock data using yahoofinance and writing it as one stock per csv
    #format is sate as index, open, high ,low, closing value and adjusted value. All values are in Japanese yen
    for stock in stocks:
        df = yf.download(str(stocks[stock]),
                      start=datestart,
                      end=dateend,
                      progress=False)
        df = df.reset_index()
        df.to_csv(str(os.getcwd()) + '/prices/' + str(stock.replace(' ','')) + '_prices.csv', index = False)


if __name__ == '__main__':
    get_stock_prices(get_nikkei225_stocks())
