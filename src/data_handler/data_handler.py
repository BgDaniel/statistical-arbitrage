import yfinance as yf
import pandas as pd
import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)


ADJ_CLOSE = "Adj Close"
CLOSE = "Close"


class DataHandler:
    """
    A class to fetch and handle historical stock data from Yahoo Finance.

    Attributes:
        tickers (list[str]): List of stock ticker symbols.
        start_date (str): Start date in "YYYY-MM-DD" format.
        end_date (str): End date in "YYYY-MM-DD" format.
        data (pd.DataFrame | None): DataFrame holding fetched adjusted close prices.
    """

    def __init__(self, tickers: list[str], start_date: str, end_date: str) -> None:
        """
        Initializes the DataHandler with tickers and date range.

        Args:
            tickers (list[str]): List of stock tickers.
            start_date (str): Start date in "YYYY-MM-DD" format.
            end_date (str): End date in "YYYY-MM-DD" format.
        """
        self.tickers: list[str] = tickers
        self.start_date: str = start_date
        self.end_date: str = end_date
        self.data: Optional[pd.DataFrame] = None

    def fetch_data(self) -> pd.DataFrame:
        """
        Fetches daily adjusted close prices for all tickers and merges them into one DataFrame.

        Returns:
            pd.DataFrame: Columns = tickers, Index = date, values = adjusted close prices.
        """
        price_data: pd.DataFrame = pd.DataFrame()

        for ticker in self.tickers:
            try:
                logger.info(f"Fetching data for {ticker}...")
                data = yf.download(
                    ticker, start=self.start_date, end=self.end_date, auto_adjust=False
                )

                # delay 1 second to avoid rate limiting / hanging
                time.sleep(1)

                # Handle MultiIndex or single-level columns
                if isinstance(data.columns, pd.MultiIndex):
                    if (ADJ_CLOSE, ticker) in data.columns:
                        series = data[ADJ_CLOSE, ticker]
                    elif (CLOSE, ticker) in data.columns:
                        series = data[CLOSE, ticker]
                    else:
                        logger.warning(
                            f"{ticker} has no 'Adj Close' or 'Close' column. Skipping."
                        )
                        continue
                else:
                    if ADJ_CLOSE in data.columns:
                        series = data[ADJ_CLOSE]
                    elif CLOSE in data.columns:
                        series = data[CLOSE]
                    else:
                        logger.warning(
                            f"{ticker} has no 'Adj Close' or 'Close' column. Skipping."
                        )
                        continue

                series = series.rename(ticker)
                price_data = pd.concat([price_data, series], axis=1)

            except Exception as e:
                logger.error(f"Error downloading {ticker}: {e}")
                continue

        self.data = price_data
        logger.info("Data fetching complete.")
        return self.data
