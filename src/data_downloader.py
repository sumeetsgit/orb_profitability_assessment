import yfinance as yf
import requests
from alpha_vantage.timeseries import TimeSeries
import pandas as pd
import os
from tqdm import tqdm
from settings import NSE_URL, NSE_EQUITY_LIST_PATH, RAW_DATA_PATH, RAW_DATA_MINUTE_PATH, ALPHAVANTAGE_API_KEY


os.makedirs(RAW_DATA_PATH, exist_ok=True)
os.makedirs(RAW_DATA_MINUTE_PATH, exist_ok=True)

# Read the tickers
tickers = pd.read_csv(NSE_EQUITY_LIST_PATH, header=None)[0].tolist()
print("Number of tickers --> {}".format(len(tickers)))


ts = TimeSeries(key=ALPHAVANTAGE_API_KEY, output_format="pandas")


def download_yfinance_data(ticker_list, rawdata_path):
    # Fetch data for each ticker
    tickers_data_available, tickers_data_unavailable, tickers_data_error = [], [], []
    for ticker in tqdm(ticker_list):
        try:
            print(f"Fetching data for {ticker}...")
            data = yf.download(f"{ticker}.NS", period="max")
            
            if not data.empty:
                # Save to CSV
                data.to_csv(f"{rawdata_path}/{ticker}.csv")
                print(f"Data saved for {ticker}")
                tickers_data_available.append(ticker)
            else:
                print(f"No data available for {ticker}")
                tickers_data_unavailable.append(ticker)
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
            tickers_data_error.append(ticker)

    print("Data download completed!")

    return tickers_data_available, tickers_data_unavailable, tickers_data_error




def download_alpha_vantage_data(ticker_list, rawdata_minute_path):
    tickers_data_available, tickers_data_unavailable, tickers_data_error = [], [], []
    for ticker in tqdm(ticker_list[:2]):
        try:
            print(f"Fetching minute data for {ticker}...")
            url = 'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={}&interval=1min&apikey={}&outputsize=full&datatype=csv'.format(ticker, ALPHAVANTAGE_API_KEY)
            # data, _ = ts.get_intraday(symbol=ticker, interval="1min", outputsize="full")
            response = requests.get(url)
            data = response.json()
            print(type(data), data)
            # data = pd.read_json(data)
            data.to_csv(f"{rawdata_minute_path}/{ticker}_minute.csv")
            print(f"Data saved for {ticker}")
            tickers_data_available.append(ticker)
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
            tickers_data_error.append(ticker)

    print("Data fetch completed!")

    return tickers_data_available, tickers_data_unavailable, tickers_data_error




def main():
    # tickers_data_available, tickers_data_unavailable, tickers_data_error = download_yfinance_data(tickers, RAW_DATA_PATH)
    tickers_data_available, tickers_data_unavailable, tickers_data_error = download_alpha_vantage_data(tickers, RAW_DATA_MINUTE_PATH)
    print("="*66)
    print("Data is available for {} tickers".format(len(tickers_data_available)))
    print("Data is unavailable for {} tickers".format(len(tickers_data_unavailable)))
    print("Data is error-prone for {} tickers --> {}".format(len(tickers_data_error), tickers_data_error))
    print("="*66)


# Execute the main function
if __name__ == "__main__":
    main()
