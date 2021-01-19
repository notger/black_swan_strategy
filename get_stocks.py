from datetime import date
import pandas as pd
import numpy as np
import yfinance as yf
from bs4 import BeautifulSoup as bs
import requests
import os

today = date.today()
cwd = os.getcwd()

def get_sp500_stocks():
    #getting all s&p500 tickers from wikipedia
    resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs(resp.text, 'lxml')
    table = soup.find('table', {'class': 'wikitable sortable'})
    stocks = []
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text[:-1]
        stocks.append(ticker.replace(".", "-"))
    return stocks


def get_stock_prices(stocks, datestart = '1980-01-01', dateend = today.strftime('%Y-%m-%d')):
    #downloading all stock data
    for stock in stocks:
        df = yf.download(stock, 
                      start=datestart, 
                      end=dateend, 
                      progress=False)
        df = df.reset_index()
        df.to_csv(str(cwd) + '/prices/' + str(stock) + '_prices.csv', index = False)

        
if __name__ == '__main__':
    get_stock_prices(get_sp500_stocks())
