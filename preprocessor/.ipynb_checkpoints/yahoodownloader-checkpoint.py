"""Contains methods and classes to collect data from
Yahoo Finance API
"""

from __future__ import annotations

import pandas as pd
import yfinance as yf
from curl_cffi import requests

# Setup session
session = requests.Session(impersonate="chrome")

class YahooDownloader:
    """Provides methods for retrieving daily stock data from
    Yahoo Finance API

    Attributes
    ----------
        start_date : str
            start date of the data (modified from neofinrl_config.py)
        end_date : str
            end date of the data (modified from neofinrl_config.py)
        ticker_list : list
            a list of stock tickers (modified from neofinrl_config.py)

    Methods
    -------
    fetch_data()
        Fetches data from yahoo API

    """

    def __init__(self, start_date: str, end_date: str, ticker_list: list):
        self.start_date = start_date
        self.end_date = end_date
        self.ticker_list = ticker_list

    def fetch_data(self, proxy=None, auto_adjust=False) -> pd.DataFrame:
        """Fetches data from Yahoo API
        Parameters
        ----------

        Returns
        -------
        `pd.DataFrame`
            7 columns: A date, open, high, low, close, volume and tick symbol
            for the specified stock ticker
        """
        # Download and save the data in a pandas DataFrame:
        data_df = pd.DataFrame()
        num_failures = 0
        for tic in self.ticker_list:
            temp_df = yf.download(
                tic,
                start=self.start_date,
                end=self.end_date,
                proxy=proxy,
                auto_adjust=auto_adjust,
                session=session
            )
            if temp_df.columns.nlevels != 1:
                temp_df.columns = temp_df.columns.droplevel(1)
            temp_df["tic"] = tic
            if len(temp_df) > 0:
                # data_df = data_df.append(temp_df)
                data_df = pd.concat([data_df, temp_df], axis=0)
            else:
                num_failures += 1
        if num_failures == len(self.ticker_list):
            raise ValueError("no data is fetched.")
        # reset the index, we want to use numbers as index instead of dates
        data_df = data_df.reset_index()
        try:
            # convert the column names to standardized names
            data_df.rename(
                columns={
                    "Date": "date",
                    "Adj Close": "adjcp",
                    "Close": "close",
                    "High": "high",
                    "Low": "low",
                    "Volume": "volume",
                    "Open": "open",
                    "tic": "tic",
                },
                inplace=True,
            )

            if not auto_adjust:
                data_df = self._adjust_prices(data_df)
        except NotImplementedError:
            print("the features are not supported currently")
        # create day of the week column (monday = 0)
        data_df["day"] = data_df["date"].dt.dayofweek
        # convert date to standard string format, easy to filter
        data_df["date"] = data_df.date.apply(lambda x: x.strftime("%Y-%m-%d"))
        # drop missing data
        data_df = data_df.dropna()
        data_df = data_df.reset_index(drop=True)
        print("Shape of DataFrame: ", data_df.shape)
        # print("Display DataFrame: ", data_df.head())

        data_df = data_df.sort_values(by=["date", "tic"]).reset_index(drop=True)

        return data_df

    def _adjust_prices(self, data_df: pd.DataFrame) -> pd.DataFrame:
        # use adjusted close price instead of close price
        data_df["adj"] = data_df["adjcp"] / data_df["close"]
        for col in ["open", "high", "low", "close"]:
            data_df[col] *= data_df["adj"]

        # drop the adjusted close price column
        return data_df.drop(["adjcp", "adj"], axis=1)

    def select_equal_rows_stock(self, df):
        df_check = df.tic.value_counts()
        df_check = pd.DataFrame(df_check).reset_index()
        df_check.columns = ["tic", "counts"]
        mean_df = df_check.counts.mean()
        equal_list = list(df.tic.value_counts() >= mean_df)
        names = df.tic.value_counts().index
        select_stocks_list = list(names[equal_list])
        df = df[df.tic.isin(select_stocks_list)]
        return df


def main():
    """
    Main script to demonstrate YahooDownloader usage
    """
    # Example parameters
    start_date = "2023-01-01"
    end_date = "2023-12-31"
    ticker_list = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]
    
    print("YahooDownloader Demo")
    print("=" * 50)
    print(f"Start Date: {start_date}")
    print(f"End Date: {end_date}")
    print(f"Tickers: {ticker_list}")
    print()
    
    try:
        # Initialize the downloader
        downloader = YahooDownloader(
            start_date=start_date,
            end_date=end_date,
            ticker_list=ticker_list,
        )
        
        # Fetch data
        print("Downloading data...")
        data = downloader.fetch_data()
        
        # Display basic information
        print(f"\nDownload completed successfully!")
        print(f"Data shape: {data.shape}")
        print(f"Date range: {data['date'].min()} to {data['date'].max()}")
        print(f"Number of unique tickers: {data['tic'].nunique()}")
        
        # Show first few rows
        print("\nFirst 10 rows of data:")
        print(data.head(10))
        
        # Show data info
        print("\nData info:")
        print(data.info())
        
        # Optional: Select equal rows
        print("\nSelecting stocks with equal data rows...")
        equal_data = downloader.select_equal_rows_stock(data)
        print(f"After selection - Data shape: {equal_data.shape}")
        print(f"Remaining tickers: {equal_data['tic'].unique()}")
        
    except Exception as e:
        print(f"Error occurred: {e}")
        print("Please check your internet connection and ticker symbols.")


if __name__ == "__main__":
    main()

