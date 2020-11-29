from alpha_vantage.timeseries import TimeSeries
from pprint import pprint
import urllib.request, json
import argparse

import pandas as pd
import datetime as dt

api_key = 'Q0ALRJ06573WNE0V'

ticker = "MSFT"
time_window = 'daily'

url_string = "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=%s&outputsize=full&apikey=%s"%(ticker,api_key)

file_to_save = '{0}_{1}.csv'.format(ticker,time_window)


ts = TimeSeries(key=api_key, output_format='pandas')
data, meta_data = ts.get_daily(symbol= ticker, outputsize='full')

#data.columns['Date', 'Open', 'High' ,'Low', 'Close','Volume']

data.rename(columns={'date': 'Date', '1. open': 'Open', '2. high': 'High', '3. low': 'Low', '4. close': 'Close', '5. volume': 'Volume' }, inplace=True)

data.sort_values('date')
data.reindex(index=data.index[::-1])
data.to_csv(file_to_save)

print('Data saved to : %s'%file_to_save) 

#df.to_csv(file_to_save)

