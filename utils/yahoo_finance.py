
from pandas_datareader import data
from os.path import join, isfile


def load_data(source, tickers, start_date, end_date):

    for ticker in tickers:
        csv_path = join('data', 'yahoo_stocks', ticker + ".csv")
        if isfile(csv_path):
            print("{}: Using preloaded data".format(ticker))
            continue
        panel_data = data.DataReader([ticker], source, start_date, end_date)

        panel_data.to_frame().to_csv(csv_path).sort_index()
        print("{}: Loaded data".format(ticker))
