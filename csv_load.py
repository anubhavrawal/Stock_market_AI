import pandas as pd

ticker = "MSFT"
time_window = 'daily'

file_to_save = '{0}_{1}.csv'.format(ticker,time_window)

df = pd.read_csv(file_to_save)

df = pd.DataFrame(df.values[::-1], df.index, df.columns)


print(df)