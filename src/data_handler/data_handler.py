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
        self.tickers: list[str] = tickers
        self.start_date: str = start_date
        self.end_date: str = end_date
        self.data: Optional[pd.DataFrame] = None

    def check_data_completeness(
        self, df: pd.DataFrame, allowed_missing_pct: float = 4.0
    ) -> None:
        """
        Checks whether the percentage of missing trading days exceeds the allowed threshold.
        Raises ValueError if data is too incomplete.

        Args:
            df (pd.DataFrame): Price DataFrame with a Date index.
            allowed_missing_pct (float): Maximum allowed percentage of missing days.
                                         Default = 5%.

        Raises:
            ValueError: If missing percentage exceeds allowed_missing_pct.
        """
        # Expected trading days (business days)
        expected_days = pd.date_range(self.start_date, self.end_date, freq="B")

        expected_count = len(expected_days)
        actual_count = len(df.index.unique())

        missing = expected_count - actual_count
        missing_pct = missing / expected_count * 100

        logger.info(f"Expected trading days: {expected_count}")
        logger.info(f"Actual trading days:   {actual_count}")
        logger.info(f"Missing days:           {missing} ({missing_pct:.2f}%)")
        logger.info(f"Allowed missing percentage: {allowed_missing_pct:.2f}%")

        if missing_pct > allowed_missing_pct:
            raise ValueError(
                f"Data completeness check failed: {missing_pct:.2f}% days missing "
                f"(allowed max = {allowed_missing_pct:.2f}%)."
            )

        elif missing_pct > 0:
            logger.warning(
                f"Data contains {missing_pct:.2f}% missing days, but within allowed limit "
                f"({allowed_missing_pct:.2f}%)."
            )
        else:
            logger.info("Data completeness check passed with no missing days.")

    # ------------------------------------------------------------
    # Fetch Data
    # ------------------------------------------------------------
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

                time.sleep(1)  # avoid rate limiting

                # MultiIndex (new yfinance) or single-index (older versions)
                if isinstance(data.columns, pd.MultiIndex):
                    if (ADJ_CLOSE, ticker) in data.columns:
                        series = data[ADJ_CLOSE, ticker]
                    elif (CLOSE, ticker) in data.columns:
                        series = data[CLOSE, ticker]
                    else:
                        logger.warning(
                            f"{ticker} missing 'Adj Close'/'Close'. Skipping."
                        )
                        continue
                else:
                    if ADJ_CLOSE in data.columns:
                        series = data[ADJ_CLOSE]
                    elif CLOSE in data.columns:
                        series = data[CLOSE]
                    else:
                        logger.warning(
                            f"{ticker} missing 'Adj Close'/'Close'. Skipping."
                        )
                        continue

                series = series.rename(ticker)
                price_data = pd.concat([price_data, series], axis=1)

            except Exception as e:
                logger.error(f"Error downloading {ticker}: {e}")
                continue

        # Perform completeness check before returning
        self.check_data_completeness(price_data)

        self.data = price_data
        logger.info("Data fetching complete.")
        return self.data
